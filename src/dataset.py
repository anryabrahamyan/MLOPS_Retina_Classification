"""
Dataset class containing the loading and preprocessing of the data
"""


import pickle
from glob import glob

import albumentations
import cv2
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from tqdm import tqdm


class RetinaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,data_folder, label_path, upsample, presaved_data_path, save_data_path, transform, image_size):
        
        self.simple_transform = albumentations.Compose([
          albumentations.Resize(image_size, image_size),
        ])

        self.transform = transform

        if presaved_data_path == None:
          label_frame = pd.read_csv(label_path)
          for image_name in glob(data_folder+"/*"):
            label_frame_index = int(image_name.split("/")[-1].split(".")[0])
            label_frame.loc[label_frame["ID"] == label_frame_index, "image_path"] = image_name
          
          label_frame = label_frame[~label_frame["image_path"].isna()]

          X_train, y_train = label_frame["image_path"].values.reshape(-1,1),label_frame.pop("Disease_Risk")
          if upsample == True:
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)

          self.data = []
          for image,label in tqdm(zip(X_train.flatten(),y_train)):
            img = cv2.imread(image)
            img = self.simple_transform(image=img)["image"]
            self.data.append((img,label))

          with open(save_data_path, "wb") as f:
            pickle.dump(self.data,f)

        else:
          with open(presaved_data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(image=self.data[idx][0])["image"],self.data[idx][1]