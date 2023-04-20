# -*- coding: utf-8 -*-
"""Capstone_Model_Test_Simple_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NC_E9zXjfAmB6haXqSWT9Ms67bNL3Znb
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install lightning
# !pip3 install ml_collections
# !pip3 install -r /content/Pytorch-RIADD/requirements.txt
# !pip3 install visdom

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !unzip /content/drive/MyDrive/CapstoneData/Test_Set.zip
# !unzip /content/drive/MyDrive/CapstoneData/Training_Set.zip
# !unzip /content/drive/MyDrive/CapstoneData/Evaluation_Set.zip

!cp /content/drive/MyDrive/CapstoneData/RFMiD_Testing_Labels.csv /content/Test_Set/
!cp /content/drive/MyDrive/CapstoneData/RFMiD_Validation_Labels.csv /content/Evaluation_Set/
!cp /content/drive/MyDrive/CapstoneData/RFMiD_Training_Labels.csv /content/Training_Set/

!git clone https://github.com/Hanson0910/Pytorch-RIADD.git
!pip install vit-pytorch

import sys
sys.path.insert(1, '/content/Pytorch-RIADD')

import albumentations
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import pickle
from glob import glob

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import urllib.request
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

image_size = 64
train_trans  = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        ToTensorV2(),
    ])

train_trans_batch  = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.MedianBlur(blur_limit = 7, p=0.3),
        albumentations.IAAAdditiveGaussianNoise(scale = (0,0.15*255), p = 0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.3),
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

train_dataset = RetinaDataset(data_folder = "/content/Training_Set/Training", label_path = "/content/Training_Set/RFMiD_Training_Labels.csv", upsample=True, presaved_data_path = "/content/drive/MyDrive/CapstoneData/train_data_upsampe.pkl", save_data_path = None, transform = train_trans_batch, image_size=64 )
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)

val_dataset = RetinaDataset(data_folder = "/content/Evaluation_Set/Validation", label_path = "/content/Evaluation_Set/RFMiD_Validation_Labels.csv", upsample=False, presaved_data_path = "/content/drive/MyDrive/CapstoneData/validation_data_upsampe.pkl", save_data_path = None, transform = valid_trans_batch, image_size=64 )
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

test_dataset = RetinaDataset(data_folder = "/content/Test_Set/Test", label_path = "/content/Test_Set/RFMiD_Testing_Labels.csv", upsample=False, presaved_data_path = "/content/drive/MyDrive/CapstoneData/test_data_upsampe.pkl", save_data_path = None, transform = valid_trans_batch, image_size=64 )
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)

# Setting the seed
L.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/content/drive/MyDrive/CapstoneData/")


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

class ViT(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

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
)
print("ViT results", results)

# !rm -rf saved_models

# Commented out IPython magic to ensure Python compatibility.
# Opens tensorboard in notebook. Adjust the path to your CHECKPOINT_PATH!

# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/CapstoneData/ViT/lightning_logs --port 6006

!git clone https://github.com/anryabrahamyan/MLOPS_Retina_Classification.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/MLOPS_Retina_Classification

!git commit -m "adding model"

!git config --global user.email "hovhannes.manushyan@gmail.com"
!git config --global user.name "HovhannesManushyan"

!git ls-remote --heads

