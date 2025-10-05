import os
import tqdm
import requests
import cv2
import numpy as np

available_models = ["neuflow_mixed", "neuflow_sintel", "neuflow_things"]

def download_model(url: str, path: str):
    print(f"Downloading model from {url} to {path}")
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024 * 1024), total=total_length // (1024 * 1024),
                               bar_format='{l_bar}{bar:10}'):
            if chunk:
                f.write(chunk)
                f.flush()


def check_model(model_path: str):
    if os.path.exists(model_path):
        return

    model_name = os.path.basename(model_path).split('.')[0]
    if model_name not in available_models:
        raise ValueError(f"Invalid model name: {model_name}")
    url = f"https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases/download/0.1.0/{model_name}.onnx"
    download_model(url, model_path)

