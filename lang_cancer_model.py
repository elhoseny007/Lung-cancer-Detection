import os
import gdown

MODEL_PATH = "lung_cancer_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=FILE_ID"
    gdown.download(url, MODEL_PATH, quiet=False)
