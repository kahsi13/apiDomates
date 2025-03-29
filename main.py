from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import requests

app = FastAPI()

# Hugging Face model bilgileri
hf_model_id = "Kahsi13/DomatesRailway"
onnx_url = f"https://huggingface.co/{hf_model_id}/resolve/main/bert_domates_model.onnx"

# ONNX model dosyasını indir (sadece ilk çalıştırmada)
onnx_path = "bert_domates_model.onnx"
if not os.path.exists(onnx_path):
    print("🔽 Hugging Face'den model indiriliyor...")
    r = requests.get(onnx_url)
    with open(onnx_path, "wb") as f:
        f.write(r.content)

# Tokenizer ve ONNX oturumu
tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
session = onnxruntime.InferenceSession(onnx_path)

# Giriş modeli
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
