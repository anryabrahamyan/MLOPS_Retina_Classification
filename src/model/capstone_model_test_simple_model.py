# -*- coding: utf-8 -*-
"""Capstone_Model_Test_Simple_Model.ipynb

Original file is located at
    https://colab.research.google.com/drive/1NC_E9zXjfAmB6haXqSWT9Ms67bNL3Znb
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install lightning
# !pip3 install ml_collections
# !pip3 install -r /content/Pytorch-RIADD/requirements_old.txt
# !pip3 install visdom

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip /content/drive/MyDrive/CapstoneData/Test_Set.zip
# !unzip /content/drive/MyDrive/CapstoneData/Training_Set.zip
# !unzip /content/drive/MyDrive/CapstoneData/Evaluation_Set.zip

# !cp /content/drive/MyDrive/CapstoneData/RFMiD_Testing_Labels.csv /content/Test_Set/
# !cp /content/drive/MyDrive/CapstoneData/RFMiD_Validation_Labels.csv /content/Evaluation_Set/
# !cp /content/drive/MyDrive/CapstoneData/RFMiD_Training_Labels.csv /content/Training_Set/

# !git clone https://github.com/Hanson0910/Pytorch-RIADD.git
# !pip install vit-pytorch

import sys

sys.path.insert(1, '/content/Pytorch-RIADD')

import os

import albumentations
import lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from data.dataset import RetinaDataset
from vision_transformer import *

# Setting the seed
L.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_size = 64


def train_model(**kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = ViT.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


train_trans = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    ToTensorV2(),
])

train_trans_batch = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.MedianBlur(blur_limit=7, p=0.3),
    albumentations.IAAAdditiveGaussianNoise(scale=(0, 0.15 * 255), p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3),
    albumentations.Cutout(max_h_size=20, max_w_size=20, num_holes=5, p=0.5),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])

valid_trans_batch = albumentations.Compose([
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])

train_dataset = RetinaDataset(data_folder="/content/Training_Set/Training",
                              label_path="/content/Training_Set/RFMiD_Training_Labels.csv", upsample=True,
                              presaved_data_path="/content/drive/MyDrive/CapstoneData/train_data_upsampe.pkl",
                              save_data_path=None, transform=train_trans_batch, image_size=64)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)

val_dataset = RetinaDataset(data_folder="/content/Evaluation_Set/Validation",
                            label_path="/content/Evaluation_Set/RFMiD_Validation_Labels.csv", upsample=False,
                            presaved_data_path="/content/drive/MyDrive/CapstoneData/validation_data_upsampe.pkl",
                            save_data_path=None, transform=valid_trans_batch, image_size=64)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

test_dataset = RetinaDataset(data_folder="/content/Test_Set/Test",
                             label_path="/content/Test_Set/RFMiD_Testing_Labels.csv", upsample=False,
                             presaved_data_path="/content/drive/MyDrive/CapstoneData/test_data_upsampe.pkl",
                             save_data_path=None, transform=valid_trans_batch, image_size=64)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/content/drive/MyDrive/CapstoneData/")

if __name__ == "__main__":
    model, results = train_model(
        model_kwargs={
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 8,
            "num_channels": 3,
            "num_patches": 64,
            "num_classes": 2,
            "dropout": 0.2,
        },
        lr=3e-4,
        train_loader=train_loader
    )
    print("ViT results", results)
