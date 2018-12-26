import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DataTrainLoader(Dataset):

    def __init__(self, path, transform):
        files_path = os.getcwd() +'/'+ path + '/'
        self.train_labels = pd.read_csv(files_path+'train_labels.csv')
        self.train_images = files_path + "train/"
        self.transform = transform

    def __create_on_hot_encoding(self, label):
        if label == 0:
            return np.array(0.0)
        elif label == 1:
            return np.array(1.0)
    
    def __getitem__(self, index):
        image = Image.open(self.train_images+self.train_labels.id[index]+'.tif')
        image = self.transform(image)
        return image, self.__create_on_hot_encoding(self.train_labels.label[index])

    def __len__(self):
        return self.train_labels.id.count()
    
class DataTestLoader(Dataset):

    def __init__(self, path, transform):
        files_path = os.getcwd() +'/'+ path + '/'
        self.test_images = files_path + "test/"
        self.onlyfiles = [f[:-4] for f in os.listdir(self.test_images)]
        self.transform = transform
    
    def __getitem__(self, index):
        image = Image.open(self.test_images+self.onlyfiles[index]+'.tif')
        image = self.transform(image)
        return image, self.onlyfiles[index]

    def __len__(self):
        return len(self.onlyfiles)