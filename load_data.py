import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import pydicom
import glob
import torch
import albumentations

class UltrasoundDataset(Dataset):
    def __init__(self,us_path,csv_path,train = True, transform=None):

        self.us_path = us_path
        self.csv_path = csv_path
        self.transform = transform
        self.train = train
        if self.train:
            self.csv = pd.read_csv(self.csv_path+'/US_train_metadata.csv')
        else:
            self.csv = pd.read_csv(self.csv_path+'/US_test_metadata.csv')

    def __getitem__(self, index):

        id,label = us_reader(self.csv,index)
        img = us_loader(self.us_path,id,self.train,self.transform)

        return id,img,label

    def __len__(self):
        return len(self.csv)


def us_reader(df,index):

    value = df.loc[index]
    return value['anonymized_id'],value['class\nbenign: 0 malignant: 1']

def us_loader(us_path,id,train,transform):
    if train:
        path = us_path+'/US_train/'+str(id)+'/ultrasound/*.dcm'
    else:
        path = us_path+'/US_test/' + str(id) + '/ultrasound/*.dcm'
    us_path = [p for p in glob.glob(path)][0]

    us_image = pydicom.dcmread(us_path).pixel_array
    resize = albumentations.Resize(256, 256)

    us_image = resize(image=us_image)['image']

    if transform:
        us_image = transform(image=us_image)['image']
    us_image = np.reshape(us_image, ((1,) + us_image.shape))


    return torch.from_numpy(us_image.copy())