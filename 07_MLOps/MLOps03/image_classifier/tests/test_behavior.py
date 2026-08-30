# tests/test_behavior.py
import pytest
import torch
import torchvision.transforms.functional as TF
from src.model import FacilityViTClassifier

# ── Fixture: 빠른 테스트를 위한 모델과 가짜 데이터 준비 ──

@pytest.fixture
def model():
    """우리가 만든 실제 ViT 모델 뼈대를 불러옵니다 (가중치는 학습 전 상태)"""
    model_instance = FacilityViTClassifier(num_classes=22)
    model_instance.eval() # 행동 테스트는 학습이 아니므로 eval() 모드 필수
    return model_instance

@pytest.fixture
def sample_batch():
    """모델이 기대하는 [Batch, Channels, Height, Width] 형태의 가짜 이미지 텐서"""
    x = torch.randn(4, 3, 224, 224) 
    y = torch.randint(0, 22, (4,))
    return x, y

# ── 행동 테스트 로직 ──

def test_invariance_horizontal_flip(model, sample_batch):
    """(1) 불변성 테스트: 좌우 반전 시 분류 예측 결과가 동일해야 함"""
    x, y = sample_batch

    # 원본 이미지 예측
    logits_original = model(x)
    preds_original = torch.argmax(logits_original, dim=1)

    # 좌우 반전 이미지 예측
    x_flipped = TF.hflip(x)
    logits_flipped = model(x_flipped)
    preds_flipped = torch.argmax(logits_flipped, dim=1)

    # 모든 예측값이 같아야 함
    assert torch.equal(preds_original, preds_flipped), (
        "좌우 반전 시 예측 카테고리가 달라집니다."
    )


def test_directional_masking(model, sample_batch):
    """(2) 방향성 테스트: 결함 부위(중앙)를 마스킹하면 위험 확률이 떨어져야 함"""
    x, y = sample_batch

    # '위험' 클래스의 인덱스가 1이라고 가정
    DANGER_CLASS_IDX = 1  

    # 원본 이미지의 위험 확률 계산
    probs_original = torch.softmax(model(x), dim=1)
    danger_probs_original = probs_original[:, DANGER_CLASS_IDX]

    # 이미지 중앙(결함 부위라고 가정)을 검게 칠함 (Masking / Cutout)
    x_masked = x.clone()
    x_masked[:, :, 100:124, 100:124] = 0.0

    # 마스킹 후 확률 계산
    probs_masked = torch.softmax(model(x_masked), dim=1)
    danger_probs_masked = probs_masked[:, DANGER_CLASS_IDX]

    # 마스킹 후에는 덜 위험해 보여야 하므로 확률값이 감소해야 정상
    assert (danger_probs_masked < danger_probs_original).all(), (
        "결함을 가렸음에도 위험 확률이 떨어지지 않았습니다."
    )