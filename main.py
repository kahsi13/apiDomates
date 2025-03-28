from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
import subprocess
import zipfile
import requests
import threading
import time

app = FastAPI()

MODEL_PATH = "bert_domates_model_quant.onnx"
TOKENIZER_PATH = "bert_domates_model_pytorch"

# Google Drive'dan dosya indirme fonksiyonu
def download_file(url, output_path):
    print(f"🔽 {output_path} indiriliyor...")
    r = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(r.content)
    print(f"✅ {output_path} indirildi.")

# Arka planda model ve tokenizer yükleme
def setup_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH):
        download_file(
            "https://drive.google.com/uc?export=download&id=1bExVJ1cuR3gAwnStUd6ceS-qyrpmzGJr",
            MODEL_PATH
        )

    if not os.path.exists(TOKENIZER_PATH):
        zip_path = "tokenizer.zip"
        download_file(
            "https://drive.google.com/uc?export=download&id=1o3xjodOVSRU-70Vg9vSQUUUP9M99L7XB",
            zip_path
        )
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TOKENIZER_PATH)
        os.remove(zip_path)

    global tokenizer, session
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = onnxruntime.InferenceSession(MODEL_PATH)

# Başlangıçta ayrı bir thread'de indirme/yükleme
threading.Thread(target=setup_model_and_tokenizer).start()

# Giriş veri yapısı
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    try:
        if "tokenizer" not in globals() or "session" not in globals():
            return {"error": "⏳ Model ve tokenizer henüz yükleniyor, lütfen birkaç saniye sonra tekrar deneyin."}

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
