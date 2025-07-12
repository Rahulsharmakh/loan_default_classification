# model_utils.py
import os
import gdown
import pickle

MODEL_FILE = "rf_final_model.pkl"
GOOGLE_DRIVE_ID = "14vCCeQodCFGmMeFs_-ocw54lKEZ_W7sl"  # replace with your real ID

def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False)
    else:
        print("Model already exists.")

def load_model():
    download_model()
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model
