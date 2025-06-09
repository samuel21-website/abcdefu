# ✅ app.py (경량화 버전 – KoGPT2 텍스트 분석만 유지)
from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import torch

app = Flask(__name__)

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
text_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
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
당신은 수질 전문가입니다. 아래 어항 상태를 분석하여 아래 형식으로 신뢰도 있고 정확하게 요약하십시오.
반복 금지, 말장난 금지, 의미 없는 문장 금지, 요점만 정리하십시오.

1️⃣ 요약 (한 문장)
2️⃣ 세부 분석 항목별 문제점 + 권장 조치 (4~5줄 이내)

- pH: {ph}, TDS: {tds}, 수온: {temp} ℃
- 질산염: {no3}, 아질산염: {no2}, 암모니아: {nh3}, 경도: {gh}
- 어종: {fish}
- 어항 크기: {tank}, 바닥재: {substrate}, 여과기: {filter_}, 기타: {others}
"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = text_model.generate(
        input_ids,
        max_length=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.3
    )
    advice = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    return render_template("result.html", advice=advice, image_result="사진 분석 기능은 제외됨.")

if __name__ == "__main__":
    app.run(debug=True)
