from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from ultralytics import YOLO
import torch
from PIL import Image
import os

app = Flask(__name__)

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
text_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)

yolo_model = YOLO("yolov8n.pt")

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
아래 어항 상태를 분석하여 다음 형식에 맞춰 전문가처럼 설명하세요.

[분석 템플릿]

1️⃣ 요약:
- 전체 상태를 한 문장으로 요약해 주세요.

2️⃣ 세부 설명:
- 수질 항목별 문제와 이유
- 생물별 위험 요소
- 권장 조치 및 개선 방법

어항 상태:
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
        repetition_penalty=1.2
    )
    advice = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    image = request.files.get("fish_image")
    image_result = "업로드된 이미지 없음"
    if image:
        path = os.path.join("static", "upload.jpg")
        image.save(path)
        results = yolo_model(path)
        labels = results[0].names
        boxes = results[0].boxes
        if boxes:
            image_result = "감지된 객체 목록:\n"
            for box in boxes:
                cls = int(box.cls[0].item())
                image_result += f"- {labels[cls]}\n"
        else:
            image_result = "감지된 객체 없음"

    return render_template("result.html", advice=advice, image_result=image_result)

if __name__ == "__main__":
    app.run(debug=True)
