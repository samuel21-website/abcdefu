# ✅ app.py (초경량 버전 – KoGPT2 제거, Hugging Face tiny GPT2 사용)
from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")
text_model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    ph = request.form.get("ph", type=float)
    tds = request.form.get("tds", type=float)
    temp = request.form.get("temp", type=float)
    fish = request.form.get("fish")
    no3 = request.form.get("no3", type=float)
    no2 = request.form.get("no2", type=float)
    nh3 = request.form.get("nh3", type=float)
    gh = request.form.get("gh", type=float)
    tank = request.form.get("tank")
    substrate = request.form.get("substrate")
    filter_ = request.form.get("filter")
    others = request.form.get("others")

    prompt = f"""
어항 상태를 요약하고 분석하시오:

1️⃣ 요약:
전체 상황을 한 문장으로 정리하십시오.

2️⃣ 세부 분석:
수질 문제와 권장 조치를 항목별로 서술하십시오.

- pH: {ph}, TDS: {tds}, 수온: {temp} ℃
- 질산염: {no3}, 아질산염: {no2}, 암모니아: {nh3}, 경도: {gh}
- 어종: {fish}
- 어항 크기: {tank}, 바닥재: {substrate}, 여과기: {filter_}, 기타: {others}
"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        input_ids,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )
    advice = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    return render_template("result.html", advice=advice, image_result="사진 분석 기능은 제외됨.")

if __name__ == "__main__":
    app.run(debug=True)
