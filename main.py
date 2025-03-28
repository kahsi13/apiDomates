from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
import subprocess
import zipfile
import requests

app = FastAPI()

# Google Drive'dan dosya indirme fonksiyonu
def download_file_from_google_drive(url, output):
    print("🔽 Dosya indiriliyor:", output)
    r = requests.get(url)
    with open(output, "wb") as f:
        f.write(r.content)

# Eğer model dosyası yoksa Google Drive'dan indir
if not os.path.exists("bert_domates_model.onnx"):
    download_file_from_google_drive(
        "https://drive.google.com/uc?export=download&id=1bExVJ1cuR3gAwnStUd6ceS-qyrpmzGJr",  # onnx dosyası
        "bert_domates_model.onnx"
    )

# Eğer tokenizer klasörü yoksa indir ve çıkar
if not os.path.exists("bert_domates_model_pytorch"):
    zip_path = "tokenizer.zip"
    download_file_from_google_drive(
        "https://drive.google.com/uc?export=download&id=1o3xjodOVSRU-70Vg9vSQUUUP9M99L7XB",  # tokenizer zip
        zip_path
    )
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("bert_domates_model_pytorch")
    os.remove(zip_path)

# Tokenizer ve ONNX modeli yükle
tokenizer = AutoTokenizer.from_pretrained("./bert_domates_model_pytorch")
session = onnxruntime.InferenceSession("bert_domates_model.onnx")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    try:
        encoding = tokenizer.encode_plus(
            input.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        ort_outs = session.run(None, ort_inputs)
        prediction = int(np.argmax(ort_outs[0]))

        return {"prediction": prediction}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
