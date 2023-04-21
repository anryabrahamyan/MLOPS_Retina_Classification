import hashlib
import os
import pytest
import sys
 
# setting path
sys.path.append('../')

from src.capstone_model_test_simple_model import *

@pytest.mark.parametrize("dataloader,shape", [(train_loader,[3,64,64]),(val_loader,[3,64,64])])
def test_dataloader_dims(dataloader,shape):
  for batch, classes in dataloader:
    for batch_elem in batch:
      assert batch_elem.shape == torch.Size(shape)

@pytest.mark.parametrize("dataloader,patchsize", [(train_loader,8),(val_loader,8)])
def test_patcher(dataloader,patchsize):
  for input_image,classes in dataloader:
    cifar_img = img_to_patch(input_image, 8, flatten_channels=False)
    B, C, H, W = input_image.shape
    assert cifar_img.shape == torch.Size([B,H*W//(patchsize**2),C,patchsize,patchsize])

@pytest.mark.parametrize("dataset_type,dataset_path", [("eval","/content/Evaluation_Set"),("train","/content/Training_Set"),("test","/content/Test_Set")])
def test_dataset_hash(dataset_type,dataset_path):
  if dataset_type == "eval":
    assert get_folder_checksum(dataset_path)=="a4437ab0c03897bb2d8a373a131c6b2b940f5497592b77f94f06fa3f97861bb5"
  elif dataset_type == "train":
    assert get_folder_checksum(dataset_path)=="1abd69acaaff5516161482eabc5aa41cf15babe3cd23d016e103931d61f09912"
  elif dataset_type == "test":
    assert get_folder_checksum(dataset_path)=="28defbb22483fee991c7a0df55061b77060e17f1e188f2c570da35e7cbd87918"