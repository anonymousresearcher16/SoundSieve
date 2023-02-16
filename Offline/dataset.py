import numpy as np
import os
from preprocess import process_audio

class AudioDataset:
    """ 
      This class works as the generator for the tf.data.dataset
    """
    def __init__(self, dataset_dir, augmentation=False, sr=5120, time=2.0, fft_len=256):
        self.file_names = []
        self.class_names = []
        self.augmentation = augmentation
        self.sr = sr
        self.time = time
        self.fft_len =256
        
        self.parse_dataset(dataset_dir)
    
    def __len__(self):
        """ Returns length of total dataset (number of datapoints) """
        return len(self.file_names)
    
    
    def __getitem__(self, idx):
        """ Return 1 sample at a time, indexed by the parameter 'idx' """
        
        """ Get the idx-th value """
        file_name = self.file_names[idx]
        class_name = self.class_names[idx]

        """ Read the audio file """
        feature = process_audio(file_name, self.augmentation, self.sr, self.time, self.fft_len)

        return feature, float(class_name)
    
    """ At each call, loop over all datapoints and yield one at a time """
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            """ Call shuffling function at the end of loop """
            if i == self.__len__() - 1:
                self.on_epoch_end()
    
    """ Shuffle manually at the end of each loop """
    def on_epoch_end(self):
        
        """ Store current values in different arrays """
        parse_file_names = self.file_names
        parse_class_names = self.class_names

        """ empty current arrays """
        self.file_names = []
        self.class_names = []

        indices = np.arange(len(parse_file_names))
        """ Shuffle indices """
        np.random.shuffle(indices)
        
        """ Read the shuffled indices and write to original arrays """
        for i in range(len(parse_file_names)):
            self.file_names.append(parse_file_names[indices[i]])
            self.class_names.append(parse_class_names[indices[i]])
            

    """ Parse each line of the metadata """
    def parse_dataset(self, dataset_dir):
        """ Temporary placeholder for the values to be initialized """
        parse_file_names = []
        parse_class_names = []

        """ Loop over each file """
        for class_name in os.listdir(dataset_dir):
          class_dir = dataset_dir + "/" + class_name
          for file_name in os.listdir(class_dir):
            file_path = class_dir + "/" + file_name
            
            """ Append values """
            parse_file_names.append(file_path)
            parse_class_names.append(class_name)
            
        """ Initialize values for generator class after shuffling indices """
        indices = np.arange(len(parse_file_names))
        np.random.shuffle(indices)
        for i in range(len(parse_file_names)):
            self.file_names.append(parse_file_names[indices[i]])
            self.class_names.append(parse_class_names[indices[i]])