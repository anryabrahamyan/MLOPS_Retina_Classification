"""
Utility functions for the training script
"""
import hashlib
import json
import os

import psutil
import requests


def get_folder_checksum(folder_path):
    """
    Returns the SHA-256 checksum hash of all the files in the given folder.
    """
    hasher = hashlib.sha256()
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, 'rb') as file:
                buf = file.read()
                hasher.update(buf)
    return hasher.hexdigest()


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


def send_data_to_elasticsearch(data):
    url = 'https://19d44e8119ca43cab9c26684a65f01fb.us-central1.gcp.cloud.es.io:443/search-mlops-retina/_doc?pipeline=ent-search-generic-ingestion'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'ApiKey {os.environ["ELASTIC_API_KEY"]}'
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.content)


def get_system_stats():
    # Get the CPU usage as a percentage
    cpu_percent = psutil.cpu_percent()

    # Get the system memory usage statistics
    mem = psutil.virtual_memory()

    # Calculate the used memory in percentage
    mem_percent = mem.percent

    # Convert memory usage to MB
    mem_mb = mem.used / 1024 / 1024

    # Return a dictionary with both CPU and memory usage
    return {"cpu_percent": cpu_percent, "mem_percent": mem_percent, "mem_mb": mem_mb}
