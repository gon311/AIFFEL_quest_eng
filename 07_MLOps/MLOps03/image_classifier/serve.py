# serve.py
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import FacilityViTClassifier

app = FastAPI(title="체육시설 안전점검 API", description="FSDL 비전 분류 프로젝트")

# ── 1. 모델 로드 (서버 시작 시 1회) ──
model = FacilityViTClassifier.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()
model.freeze()

# 워크시트 기획: 출력은 "정상 / 수리대상 / 교체폐기 대상"
# (참고: 만약 모델이 22개 클래스로 학습되었다면, 22개를 이 3가지 범주로 매핑하는 코드가 필요합니다. 
# 여기서는 이해하기 쉽게 처음부터 3개 클래스로 학습했다고 가정합니다.)
LABELS = ['정상', '수리대상', '교체폐기 대상']

# ── 2. 추론용 전처리 세팅 (DataModule의 Val 세팅과 동일해야 함) ──
infer_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── 3. API 응답 형식 정의 ──
class InferenceResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict

# ── 4. 엔드포인트 정의 (워크시트 기획: POST /classifier) ──
@app.post("/classifier", response_model=InferenceResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    사용자가 업로드한 이미지를 받아 안전점검 상태를 분류합니다.
    """
    # 1. 파일 읽기 및 이미지 변환
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    # 2. 전처리 (Resize & Normalize)
    transformed = infer_transform(image=image_np)
    # 모델은 [Batch, Channels, Height, Width]를 원하므로 앞에 배치 차원(1)을 추가해줌
    pixel_values = transformed["image"].unsqueeze(0) 
    
    # 3. 모델 추론
    with torch.no_grad():
        outputs = model(pixel_values) 
        logits = outputs.logits  # Hugging Face 모델 출력에서 logits 추출
        probs = torch.softmax(logits, dim=-1)[0] # 확률값으로 변환

    # 4. 결과 포맷팅 및 반환
    pred_idx = probs.argmax().item()
    return InferenceResponse(
        category=LABELS[pred_idx],
        confidence=round(probs[pred_idx].item(), 4),
        all_scores={LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))},
    )