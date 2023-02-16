import librosa
import numpy as np
import random
from preprocess import truncate_audio, fix_audio_length
import tensorflow as tf
from model import create_model
import os
import sklearn
from sklearn.linear_model import Ridge

DATASET_NAME = "ESC-kitchen"

def create_perforated_feature(samples, sample_rate, perforation_mask, fft_len=256, \
                                hop_len=128, division=0.1, time=2):
    sample_each_window = int(division * sample_rate)
    time_step = int(time / division)
    
    spectrogram = np.zeros((perforation_mask.shape[0], 3*time_step, 128))
    for j in range(perforation_mask.shape[0]):
        mask = perforation_mask[j]
        
        for i in range(time_step):
            if mask[i] == 1:
                sample_i = samples[i*sample_each_window : (i+1)*sample_each_window]
                stft = librosa.stft(sample_i, n_fft=fft_len, hop_length=hop_len, \
                                    win_length=fft_len, center=False)
                stft = stft[:-1, :]
                stft = stft.T
                D = np.abs(stft)
                D = D ** 2
                spectrogram[j, i*3 : (i+1)*3, :] = D
    
    for j in range(perforation_mask.shape[0]):
        mask = perforation_mask[j]
        start = -1
        end = -1
        for i in range(time_step):
            if mask[i] == 0:
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
                    prev_spectrogram = spectrogram[j, (start-1)*3:start*3, :]
                    next_spectrogram = spectrogram[j, (end+1)*3:(end+2)*3, :]
                    blocks_to_impute = (end - start +1)
                    current_time = start
                    while blocks_to_impute != 0:
                        ratio = (current_time - (start - 1)) / ((end + 1) - (start - 1))
                        if prev_spectrogram.shape[0] == 0:
                            spectrogram[j, current_time*3: (current_time+1)*3, :] = next_spectrogram
                        elif next_spectrogram.shape[0] == 0:
                            spectrogram[j, current_time*3: (current_time+1)*3, :] = prev_spectrogram
                        else:
                            spectrogram[j, current_time*3: (current_time+1)*3, :] = (1-ratio)*prev_spectrogram + ratio*next_spectrogram
                        current_time += 1
                        blocks_to_impute -= 1
                        start = -1
        if start != -1:
            prev_spectrogram = spectrogram[j, (start-1)*3:start*3, :]
            
            while start < time_step:
                spectrogram[j, start*3: (start+1)*3, :] = prev_spectrogram
                start += 1

    return spectrogram

def create_perforation_mask_for_importance(start, length, time=2, division=0.1):
    time_step = int(time / division)
    #perforation_mask = random_state.randint(0, 2, num_samples * time_step)\
    #                    .reshape((num_samples, time_step))

    perforation_mask = np.ones((1, time_step))
    perforation_mask[0, start+1: start+length+1] = 0
    
    return perforation_mask

def create_perforation_mask_for_lime(num_samples=5000, time=2, division=0.1):
    time_step = int(time / division)
    random_state = np.random.mtrand._rand
    perforation_mask = random_state.randint(0, 2, num_samples * time_step)\
                       .reshape((num_samples, time_step))

    perforation_mask[0, :] = 1
    for j in range(perforation_mask.shape[0]):
        if np.count_nonzero(perforation_mask[j]) == 0:
                for k in range(time_step):
                    if random.random() > 0.5:
                        perforation_mask[j][k] = 1
    return perforation_mask
    

def read_audio_file(file_name, target_sr=5120, time=2):
    samples, sr = librosa.load(file_name)

    samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    window_len = int(sr//10)
    truncated_samples = truncate_audio(samples, window_len=window_len)
    if truncated_samples.shape[0] == 0:
        print(file_name)
    fixed_length_samples = fix_audio_length(truncated_samples, sr, time)

    return fixed_length_samples

def get_model(input_shape, num_classes, dataset_name):
    model = create_model(input_shape, num_classes)
    model_name = dataset_name + "/checkpoints3/model_test"
    model.load_weights(model_name + "_weights")
    return model


def infer(features, target, model, features_layer1, bs=64, num_classes=9):
    dataset = tf.data.Dataset.from_tensor_slices((features)).batch(bs)
    
    labels = np.zeros((features.shape[0], num_classes))
    features_1 = np.zeros((1, 90, 32))
    count = 0
    for batch_inputs in dataset:
        features = batch_inputs
        predicted_labels = model(features, training=False)   
        
        labels[count:count+features.shape[0], :] = predicted_labels
        if count==0:
            f = features_layer1(features, training=False) 
            features_1[0, :, :] = f[0, :, :, 0]
        count += features.shape[0]
    return labels, features_1

def test():
    #file_name = "Urban/Dataset/Train/air_conditioner/30204-0-0-11.wav"
    #file_name = "Urban/Dataset/Train/air_conditioner/57320-0-0-33.wav"
    root_dir = DATASET_NAME + "/Dataset/Train/"
    fft_len = 256
    hop_len = int(fft_len // 2)
    sr = 5120
    time=2
    division = 0.1
    input_shape = (60, 128, 1)
    num_classes=9
    batch_size = 70
    time_step = int(time / division)
    prediction_length = 5
    num_samples = (time_step-prediction_length-1) * prediction_length + 1
    class_to_idx = {'air_conditioner':0, 'car_horn': 1, 'children_playing': 2, 'dog_bark': 3, 'drilling': 4, \
                    'engine_idling': 5, 'jackhammer': 6, 'siren': 7, 'street_music': 8}

    train_x = np.zeros((0, 3, 32, 1))
    train_y = np.zeros((0, prediction_length))

    model = get_model(input_shape, num_classes)
            
    features_layer1 = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(name="conv2d").output
    )
    
    for class_name in os.listdir(root_dir):
        class_path = root_dir + class_name
        target = class_to_idx[class_name]
        for file_name in os.listdir(class_path):
            file_path = class_path + "/" + file_name

            samples = read_audio_file(file_path, sr, time)
            perf_features = np.zeros((num_samples, int(time_step*3), 128, 1))

            perforation_mask = create_perforation_mask_for_importance(0, -1, time, division)
            perforated_features = create_perforated_feature(samples, sr, perforation_mask, fft_len, hop_len, division, time)
            perforated_features = perforated_features[:, :, :, np.newaxis]
            perf_features[0, :, :, :] = perforated_features[0, :, :, :]

            
            count = 1
            for i in range(0, time_step-prediction_length-1):
                for j in range(prediction_length):
                    perforation_mask = create_perforation_mask_for_importance(i, j, time, division)
                    perforated_features = create_perforated_feature(samples, sr, perforation_mask, fft_len, hop_len, division, time)
                    perforated_features = perforated_features[:, :, :, np.newaxis]
                    perf_features[count, :, :, :] = perforated_features[0, :, :, :]
                    count += 1
                    
            
            
            labels, features_1 = infer(perf_features, target, model, features_layer1, batch_size)
            features_1 = features_1[0, :, :]
            
            original_label = labels[0]
            count = 1
            X = np.zeros((time_step-prediction_length-1, 3, 32, 1))
            y = np.zeros((time_step-prediction_length-1, prediction_length))

            for i in range(0, time_step-prediction_length-1):
                prediction_length_labels = np.zeros(prediction_length)
                for j in range(prediction_length):
                    label_j = labels[count]
                    prediction_length_labels[j] = (original_label - label_j)
                    count += 1
                y[i, :] = prediction_length_labels
                X[i, :, :, 0] = features_1[i*3:(i+1)*3, :]
            
            train_x = np.concatenate((train_x, X), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    training_data = {'X': train_x, 'y': train_y}

    np.save('importance_training_data.npy', training_data, allow_pickle=True)
    print(train_x.shape, train_y.shape)

def kernel(d, kernel_width=.25):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

def test_lime(dataset_name, num_classes):
    root_dir = dataset_name + "/Dataset/Train/"
    fft_len = 256
    hop_len = int(fft_len // 2)
    sr = 5120
    time=3
    division = 0.1
    input_shape = (90, 128, 1)
    batch_size = 70
    time_step = int(time / division)
    prediction_length = 5
    num_samples = 5000

    train_x = np.zeros((0, 3, 32, 1))
    train_y = np.zeros((0, prediction_length))

    model = get_model(input_shape, num_classes, dataset_name)
            
    features_layer1 = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(index=0).output
    )
    
    perforation_mask = create_perforation_mask_for_lime(num_samples, time, division)
    for class_name in os.listdir(root_dir):
        class_path = root_dir + class_name
        target = int(class_name)
        
        
        for file_name in os.listdir(class_path):
            file_path = class_path + "/" + file_name
            file_name = dataset_name + '/perforation_masks_train3/' + class_name+'_'+file_name.split('.')[0]+'.npy'
            if os.path.exists(file_name):
                continue
            samples = read_audio_file(file_path, sr, time)
            perf_features = np.zeros((num_samples, int(time_step*3), 128, 1))

            random_state = np.random.mtrand._rand
            
            perforated_features = create_perforated_feature(samples, sr, perforation_mask, fft_len, hop_len, division, time)
            perforated_features = perforated_features[:, :, :, np.newaxis]
            perf_features[:, :, :, :] = perforated_features[:, :, :, :]

            labels, features_1 = infer(perf_features, target, model, features_layer1, batch_size, num_classes)
            features_1 = features_1[0, :, :]

            t_labels = labels

            labels = labels[:, target]
            labels = np.array(labels)

            distances = sklearn.metrics.pairwise_distances(perforation_mask, perforation_mask[0].reshape(1, -1), metric='cosine').ravel()
            weights = kernel(distances)

            labels_column = labels
            features = np.array(range(perforation_mask.shape[1]))

            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=random_state)
            model_regressor.fit(perforation_mask, labels_column, sample_weight=weights)

            explanation = zip(features, model_regressor.coef_)
            #explanation =sorted(zip(features, model_regressor.coef_), key=lambda x: x[1], reverse=True)
            expl= dict(explanation)
            
            mask_data = {'mask': perforation_mask, 'labels': t_labels, 'explanation': expl}
           
            np.save(file_name, mask_data, allow_pickle=True)
            '''
            X = np.zeros((time_step-prediction_length, 3, 32, 1))
            y = np.zeros((time_step-prediction_length, prediction_length))

            
            for i in range(0, time_step-prediction_length):
                prediction_length_labels = np.zeros(prediction_length)
                for j in range(1, prediction_length+1):
                    index = i + j
                    rank = expl[index]
                    prediction_length_labels[j-1] = rank
                    
                y[i, :] = prediction_length_labels
                X[i, :, :, 0] = features_1[i*3:(i+1)*3, :]
                
            train_x = np.concatenate((train_x, X), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)

            #mask_data = {'mask': perforation_mask, 'labels': labels, 'explanation': expl}
            #file_name = dataset_name + '/perforation_masks_test/' + class_name+'_'+file_name.split('.')[0]+'.npy'
            #np.save(file_name, mask_data, allow_pickle=True)
            
    training_data = {'X': train_x, 'y': train_y}

    np.save(dataset_name + '/importance_training_data3 .npy', training_data, allow_pickle=True)
    '''
#test()
#test_lime(dataset_name="Urban", num_classes=9)
#test_lime(dataset_name="ESC-animal", num_classes=11)
#test_lime(dataset_name="ESC-indoor", num_classes=9)
test_lime(dataset_name="ESC-kitchen", num_classes=8)