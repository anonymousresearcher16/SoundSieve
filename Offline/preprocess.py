import librosa
import numpy as np
import random
import math
from model import create_model
import tensorflow as tf
from importance_predictor import get_predictor_model


def get_model_layer_1(input_shape=(60,128,1), num_classes=9, dataset="Urban"):
    model = create_model(input_shape, num_classes)
    model_name = dataset + "/checkpoints/model_test"
    model.load_weights(model_name + "_weights")

    layer1_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name="conv2d").output
    )
    return model, layer1_model


def truncate_audio(samples, window_len=256):
    sum_amplitudes = []
    
    for i in range(samples.shape[0] - window_len):
        start_i = i
        end_i = i + window_len

        sample_i = samples[start_i: end_i]
        sum_amplitudes.append(np.sum((sample_i) ** 2))

    twenty_percentile = np.percentile(sum_amplitudes, 10)

    start = -1
    end = -1

    for i in range(samples.shape[0] - window_len):
        start_i = i
        end_i = i + window_len
        sample_i = samples[start_i: end_i]
        sum_power = np.sum((sample_i) ** 2)

        if np.abs(sum_power - twenty_percentile) > 0.001 and start == -1:
            start = start_i
            break

    i = samples.shape[0] - 1

    while i > window_len:
        start_i = i - window_len
        end_i = i

        sample_i = samples[start_i: end_i]
        sum_power = np.sum((sample_i) ** 2)

        if np.abs(sum_power - twenty_percentile) > 0.001 and end == -1:
            end = end_i
            break

        i -= 1

    samples_truncated = samples[start:end]
    
    return samples_truncated


def fix_audio_length(samples, sr, time):
    length = int(time * sr)
    if length < len(samples):
        #start = random.randint(0, len(samples) - length)
        start = 0
        new_sample = samples[start:start + length]
    elif length > len(samples):
        new_sample = np.pad(samples, (0, length - len(samples)), "constant")
    else:
        new_sample = samples
    
    return new_sample


def compute_spectrogram(samples, sr, fft_len, time=2, division=0.1):
    time_step = int(time / division)
    sample_each_window = int(division * sr)

    spectrogram = np.zeros((0, 128))
    for i in range(time_step):
        sample_i = samples[i*sample_each_window : (i+1)*sample_each_window]
        stft = librosa.stft(sample_i, n_fft=fft_len, hop_length=int(fft_len//2), \
                                    win_length=fft_len, center=False)
        stft = stft[:-1, :]
        stft = stft.T
        spectrogram = np.concatenate((spectrogram, stft), axis=0)
    D = np.abs(spectrogram)
    D = D ** 2
    
    #stft = librosa.stft(samples, n_fft=fft_len, hop_length=int(fft_len//2), win_length=fft_len)
    #stft = stft[:-1,:-1]
    #stft = stft.T
    #D = np.abs(stft)
    #D = D ** 2

    #max_val = np.max(D)
    #min_val = np.min(D)
    
    #D = (D-min_val)/(max_val-min_val)
    
    D = D[:, :, np.newaxis]
    return D

def create_uniform_mask(time=2, division=0.1, budget_block = 5, charge_time_to_add_block = 1.5):
    time_step = int(time / division)
    perforation_mask = np.ones(time_step)
    
    block_to_skip = int(math.ceil(budget_block * charge_time_to_add_block))
    i = 0
        
    while i < time_step:
        i += budget_block
        if i + block_to_skip >= time_step:
            perforation_mask[i : ] = 0
        else:
            perforation_mask[i : i + block_to_skip] = 0
        i += block_to_skip
    
    return perforation_mask

def create_random_mask(time, division=0.1, keep_percentage=0.5):
    time_step = int(time / division)
    perforation_mask = np.ones(time_step)
    i = 0
        
    while i < time_step:
        if(random.random() > keep_percentage):
            perforation_mask[i] = 0
        i += 1
    perforation_mask[0] = 1
    return perforation_mask

def informative_perforation_mask(samples, sample_rate, fft_len=256, hop_len=128, division=0.1, time=2, prediction_length=5, budget_block = 5, charge_time_to_add_block = 1):
    sample_each_window = int(division * sample_rate)
    time_step = int(time / division)
    perforation_mask = np.ones(time_step)
    threshold = 1e-5
    second_threshold = 1e-3
    
    spectrogram = np.zeros((3*time_step, 128))
    for i in range(time_step):
        sample_i = samples[i*sample_each_window : (i+1)*sample_each_window]
        
        stft = librosa.stft(sample_i, n_fft=fft_len, hop_length=hop_len, \
                                    win_length=fft_len, center=False)
        stft = stft[:-1, :]
        stft = stft.T
        D = np.abs(stft)
        D = D ** 2
        spectrogram[i*3 : (i+1)*3, :] = D
    
    spectrogram = spectrogram[np.newaxis, :, :, np.newaxis]
    model, layer_1_model = get_model_layer_1()
    predictor_model = get_predictor_model(input_shape=(3,32,1), prediction_length=5, load=True)
    dataset = tf.data.Dataset.from_tensor_slices((spectrogram)).batch(1)
    for batch_inputs in dataset:
        features = batch_inputs
        f = layer_1_model(features, training=False)  
    
    current_budget = budget_block

    i = 0
    while i <= time_step-prediction_length-1:
        current_budget -= 1
        current_feature = f[:, i*3:(i+1)*3, :, :]
        importance_prediction = predictor_model(current_feature, training=False)
        
        
        if i == time_step-prediction_length-1:
            for j in range(prediction_length):
                if importance_prediction[0, j] < threshold:
                    continue
                else:
                    break
            new_budget = current_budget + (j)/charge_time_to_add_block
            perforation_mask[i+1: i+j+1] = 0
            count = 0
            while math.floor(new_budget) != 0:
                new_budget -= 1
                count += 1
            perforation_mask[i+j+count:] = 1
            break
        
        else:
            for j in range(prediction_length):
                if importance_prediction[0, j] < threshold:
                    continue
                else:
                    new_budget = current_budget + (j)/charge_time_to_add_block
                    if math.floor(new_budget) >= 2:
                        perforation_mask[i+1: i+j+1] = 0
                        break
                    else:
                        if math.floor(new_budget) >= 1:
                            if importance_prediction[0, j] < second_threshold:
                                continue
                            else:
                                
                                perforation_mask[i+1: i+j+1] = 0
                                break
                        else:
                            continue
            i += (j+1)
            current_budget += (j)/charge_time_to_add_block
            
        
    return perforation_mask
    

def global_informative_perforation_mask(importance, time=2.0, division=0.1, budget_block=6, charge_time_to_add_block = 1):
    time_step = int(time / division)
    perforation_mask = np.zeros(time_step)
    percentage = int(time_step * 1 / (1+charge_time_to_add_block))

    for i in range(percentage):
        perforation_mask[importance[i]] = 1
    
    current_budget = budget_block
    i = 0
    while i < perforation_mask.shape[0]:
        if perforation_mask[i] == 1:
            if np.floor(current_budget) == 0:
                perforation_mask[i] = 0
                current_budget += 1.0/ charge_time_to_add_block
            else:
                current_budget -= 1
        elif perforation_mask[i] == 0:
            if np.floor(current_budget) != 0:
                chargable = True
                for j in range(i+1, i+1+int(np.ceil(charge_time_to_add_block))):
                    if j >= perforation_mask.shape[0] - 1:
                        chargable = False
                        break
                    if perforation_mask[j] == 1:
                        chargable = False
                if chargable:
                    perforation_mask[i] = 1
                    current_budget -= 1
                else:
                    if current_budget < budget_block:
                        current_budget += 1.0/ charge_time_to_add_block
            else:
                current_budget += 1.0/ charge_time_to_add_block
        i += 1

    return perforation_mask


def compute_spectrogram_perforated(samples, sample_rate, perforation_mask, fft_len=256, \
                                hop_len=128, division=0.1, time=2):
    sample_each_window = int(division * sample_rate)
    time_step = int(time / division)
    
    spectrogram = np.zeros((3*time_step, 128))

    for i in range(time_step):
        if perforation_mask[i] == 1:
            sample_i = samples[i*sample_each_window : (i+1)*sample_each_window]
            stft = librosa.stft(sample_i, n_fft=fft_len, hop_length=hop_len, \
                                    win_length=fft_len, center=False)
            stft = stft[:-1, :]
            stft = stft.T
            D = np.abs(stft)
            D = D ** 2
            spectrogram[i*3 : (i+1)*3, :] = D
    
    start = -1
    end = -1
    for i in range(time_step):
        if perforation_mask[i] == 0:
            if start == -1:
                start = i
                end = i
            else:
                end = i
        else:
            if start == -1:
                continue
            else:
                blocks_to_impute = (end - start + 1)
                prev_spectrogram = spectrogram[(start-1)*3:start*3, :]
                next_spectrogram = spectrogram[(end+1)*3:(end+2)*3, :]
                blocks_to_impute = (end - start +1)
                current_time = start
                while blocks_to_impute != 0:
                    ratio = (current_time - (start - 1)) / ((end + 1) - (start - 1))
                    if prev_spectrogram.shape[0] == 0:
                        spectrogram[current_time*3: (current_time+1)*3, :] = next_spectrogram
                    elif next_spectrogram.shape[0] == 0:
                        spectrogram[current_time*3: (current_time+1)*3, :] = prev_spectrogram
                    else:
                        spectrogram[current_time*3: (current_time+1)*3, :] = (1-ratio)*prev_spectrogram + ratio*next_spectrogram
                    current_time += 1
                    blocks_to_impute -= 1
                    start = -1
                    #ratio = (current_time - (start - 1)) / ((end + 1) - (start - 1))
                    #spectrogram[current_time*3: (current_time+1)*3, :] = (1-ratio)*prev_spectrogram + ratio*next_spectrogram
                    #current_time += 1
                    #blocks_to_impute -= 1
    if start != -1:
        prev_spectrogram = spectrogram[(start-1)*3:start*3, :]
        while start < time_step:
            spectrogram[start*3: (start+1)*3, :] = prev_spectrogram
            start += 1
            
    spectrogram = spectrogram[:, :, np.newaxis]
    return spectrogram


def compute_spectrogram_perforated_other(samples, sample_rate, perforation_mask, fft_len=256, \
                                hop_len=128, division=0.1, time=2):
    sample_each_window = int(division * sample_rate)
    time_step = int(time / division)
    
    spectrogram = np.zeros((3*time_step, 128))

    for i in range(time_step):
        if perforation_mask[i] == 1:
            sample_i = samples[i*sample_each_window : (i+1)*sample_each_window]
            stft = librosa.stft(sample_i, n_fft=fft_len, hop_length=hop_len, \
                                    win_length=fft_len, center=False)
            stft = stft[:-1, :]
            stft = stft.T
            D = np.abs(stft)
            D = D ** 2
            spectrogram[i*3 : (i+1)*3, :] = D
        else:
            spectrogram[i*3 : (i+1)*3, :] = 0
    spectrogram = spectrogram[:, :, np.newaxis]

    return spectrogram

def process_audio(file_name, augmentation, target_sr=5120, time=2.0, fft_len=256):
    samples, sr = librosa.load(file_name)
    
    samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    window_len = int(sr//10)
    truncated_samples = truncate_audio(samples, window_len=window_len)
    if truncated_samples.shape[0] == 0:
        print(file_name)
    fixed_length_samples = fix_audio_length(truncated_samples, sr, time)
    
    if augmentation:
        #importance = [10, 6,  8, 12,  4,  9,  7,  2, 14,  5, 11,  3, 13, 15, 16,  1,  0, 17, 18, 19]  #Urban
        #importance = [ 1,  6,  4,  7,  0,  8, 12,  3,  2, 11, 10,  5, 13, 14,  9, 16, 15, 18, 17, 19]  #Kitchen
        #importance = [ 5,  2,  8,  1,  6,  7, 10,  4,  3, 14,  9, 12, 13,  0, 11, 15, 16, 17, 19, 18]  #indoor
        #importance = [ 5,  7, 11,  6, 10, 13, 12,  8,  9,  4, 14,  2,  3, 15,  1, 16, 17,  0, 18, 19]  #animal

        importance = [0,6,5,4,1,8,2,3,7,10,9,11,12,21,13,22,16,14,18,19,17,26,25,20,28,15,24,29,23,27]

        #mask = global_informative_perforation_mask(importance, time, division=0.1, budget_block=5, charge_time_to_add_block = 3)
        
        mask = create_uniform_mask(time, division=0.1, budget_block=5, charge_time_to_add_block=3)
        #mask = create_random_mask(time, division=0.1, keep_percentage=0.5)
        s="["
        for i in range(mask.shape[0]):
            s += str(int(mask[i]))
            s += ", "
        s+="]"
        print(s)
        exit(0)
        
        spectrogram = compute_spectrogram_perforated_other(fixed_length_samples, sr, mask, fft_len, hop_len=int(fft_len//2), division=0.1, time=time)
        #spectrogram = compute_spectrogram_perforated(fixed_length_samples, sr, mask, fft_len, hop_len=int(fft_len//2), division=0.1, time=time)
    else:
        spectrogram = compute_spectrogram(fixed_length_samples, sr, fft_len, time)

    return spectrogram


def process_audio_from_mask(file_name, mask, target_sr=5120, time=2.0, fft_len=256):
    samples, sr = librosa.load(file_name)

    samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    window_len = int(sr//10)
    truncated_samples = truncate_audio(samples, window_len=window_len)
    if truncated_samples.shape[0] == 0:
        print(file_name)
    fixed_length_samples = fix_audio_length(truncated_samples, sr, time)
    
    #mask = create_random_mask(time, division=0.1, keep_percentage=0.5)
    spectrogram = compute_spectrogram_perforated(fixed_length_samples, sr, mask, fft_len, hop_len=int(fft_len//2), division=0.1, time=time)
    

    return spectrogram

