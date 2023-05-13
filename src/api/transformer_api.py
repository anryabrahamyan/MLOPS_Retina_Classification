import pickle
import time

import gdown
import lightning as L
import torch
from fastapi import FastAPI, File
from model.final_model import ViT
from utils import send_data_to_elasticsearch, get_system_stats

L.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

fd = "1-hWJi8R9D2rletbjTiCd8G3R9TVTKtYQ"
gdown.download(id=fd, output="capstone_model.ckpt", quiet=False)

kwargs = {
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "patch_size": 8,
    "num_channels": 3,
    "num_patches": 64,
    "num_classes": 2,
    "dropout": 0.2,
}
model = ViT(kwargs, lr=3e-4)
model = ViT.load_from_checkpoint("capstone_model.ckpt", map_location=torch.device('cpu'))

app = FastAPI(docs_url="/swagger")


@app.post("/")
async def create_file(file: bytes = File()):
    sample_data = pickle.loads(file)
    results = {"model_preds": str(model(sample_data))}
    system_stats = get_system_stats()
    logs = results.copy()
    logs["time"] = time.time()
    logs["file_size"] = len(file)
    logs.update(system_stats)
    send_data_to_elasticsearch(logs)

    return results
