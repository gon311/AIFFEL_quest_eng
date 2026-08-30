import pytest
import torch
import torchvision.transforms.functional as TF

# (실제 프로젝트에서는 직접 만든 모델 클래스를 import)
# from src.model import FacilityViTClassifier


@pytest.fixture
def mock_batch():
    """테스트용 가짜 이미지 배치 (Batch, C, H, W)와 22개 클래스 레이블"""
    # 224x224 RGB 이미지 4장 (임의의 텐서)
    x = torch.randn(4, 3, 224, 224)
    # 0~21 사이의 정수 레이블 4개
    y = torch.randint(0, 22, (4,))
    return x, y


@pytest.fixture
def mock_model():
    """초기화된 가짜 비전 모델 (실제 구현체로 대체하여 테스트)"""
    # 22개 클래스를 출력하는 임시 신경망 세팅
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 22))
    return model


# ---------------------------------------------------------
# 1. 데이터 기댓값 테스트 (Data Expectation Tests) 3개
# ---------------------------------------------------------
def test_data_shape(mock_batch):
    """(1) 텐서 차원 확인: [Batch, 3, 224, 224]"""
    x, y = mock_batch
    assert x.ndim == 4, "이미지 텐서는 4차원이어야 합니다."
    assert x.shape[1:] == (3, 224, 224), (
        f"예상 형태 [3, 224, 224], 실제 형태 {x.shape[1:]}"
    )


def test_data_normalization_bounds(mock_batch):
    """(2) 정규화 범위 확인: 극단적인 이상치(255 등)가 없어야 함"""
    x, _ = mock_batch
    # 느슨한 경계 적용 (-5.0 ~ 5.0)
    assert x.min() >= -5.0 and x.max() <= 5.0, (
        "픽셀 값이 정규화 범위를 벗어났습니다. 전처리 누락을 확인하세요."
    )


def test_label_validity(mock_batch):
    """(3) 레이블 유효성: 결측치 없고 0~21 사이 정수"""
    _, y = mock_batch
    assert not torch.isnan(y.float()).any(), "레이블에 결측치(NaN)가 있습니다."
    assert (y >= 0).all() and (y <= 21).all(), "레이블 값이 0~21 범위를 벗어났습니다."
    assert y.dtype in [torch.int32, torch.int64], "레이블은 정수형이어야 합니다."


# ---------------------------------------------------------
# 2. 암기 테스트 (Overfit Test) 1개
# ---------------------------------------------------------
def test_overfit_single_batch(mock_model, mock_batch):
    """(4) 단일 배치 암기 테스트: 1 Step 후 Loss가 감소하는지 확인"""
    x, y = mock_batch
    mock_model.train()  # 학습 모드
    optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-2)  # 높은 LR

    # 1 Step 전 Loss
    logits_before = mock_model(x)
    loss_before = torch.nn.functional.cross_entropy(logits_before, y).item()

    # 학습 1 Step 진행
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(mock_model(x), y)
    loss.backward()
    optimizer.step()

    # 1 Step 후 Loss
    logits_after = mock_model(x)
    loss_after = torch.nn.functional.cross_entropy(logits_after, y).item()

    assert loss_after < loss_before, (
        f"암기 실패: Loss가 감소하지 않음 ({loss_before:.4f} -> {loss_after:.4f})"
    )


# ---------------------------------------------------------
# 3. 행동 테스트 (Behavioral Tests) 2개
# ---------------------------------------------------------
def test_invariance_horizontal_flip(mock_model, mock_batch):
    """(5) 불변성 테스트: 좌우 반전 시 분류 예측 결과가 동일해야 함"""
    x, y = mock_batch
    mock_model.eval()

    # 원본 예측
    logits_original = mock_model(x)
    preds_original = torch.argmax(logits_original, dim=1)

    # 좌우 반전 이미지 예측 (torchvision 활용)
    x_flipped = TF.hflip(x)
    logits_flipped = mock_model(x_flipped)
    preds_flipped = torch.argmax(logits_flipped, dim=1)

    # 모든 예측값이 같아야 함
    assert torch.equal(preds_original, preds_flipped), (
        "좌우 반전 시 예측 카테고리가 달라집니다."
    )


def test_directional_masking(mock_model, mock_batch):
    """(6) 방향성 테스트: 결함 부위(중앙)를 마스킹하면 위험 확률이 떨어져야 함"""
    x, y = mock_batch
    mock_model.eval()

    DANGER_CLASS_IDX = 1  # '위험' 클래스의 인덱스라고 가정

    # 원본 이미지의 위험 확률 계산
    probs_original = torch.softmax(mock_model(x), dim=1)
    danger_probs_original = probs_original[:, DANGER_CLASS_IDX]

    # 이미지 중앙(결함 부위라고 가정)을 검게 칠함 (Masking / Cutout)
    x_masked = x.clone()
    x_masked[:, :, 100:124, 100:124] = 0.0

    # 마스킹 후 확률 계산
    probs_masked = torch.softmax(mock_model(x_masked), dim=1)
    danger_probs_masked = probs_masked[:, DANGER_CLASS_IDX]

    # 마스킹 후에는 위험 확률값이 감소해야 정상
    assert (danger_probs_masked < danger_probs_original).all(), (
        "결함을 가렸음에도 위험 확률이 떨어지지 않았습니다."
    )
