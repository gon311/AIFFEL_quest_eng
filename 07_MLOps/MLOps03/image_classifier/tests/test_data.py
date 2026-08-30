# tests/test_data.py
import pytest
import torch

# ── Fixture: 데이터 검증을 위한 가짜 배치 ──
# (실무 팁: 실제로는 FacilityDataModule을 불러와서 진짜 이미지 배치를 1개 뽑아 쓰는 것이 좋습니다!)
@pytest.fixture
def sample_batch():
    """테스트용 가짜 이미지 배치 (Batch, C, H, W)와 22개 클래스 레이블"""
    # 224x224 RGB 이미지 4장 
    x = torch.randn(4, 3, 224, 224)
    # 0~21 사이의 정수 레이블 4개
    y = torch.randint(0, 22, (4,))
    return x, y

# ---------------------------------------------------------
# 데이터 기댓값 테스트 (Data Expectation Tests) 3개
# ---------------------------------------------------------

def test_data_shape(sample_batch):
    """(1) 텐서 차원 확인: [Batch, 3, 224, 224]"""
    x, y = sample_batch
    assert x.ndim == 4, "이미지 텐서는 4차원이어야 합니다."
    assert x.shape[1:] == (3, 224, 224), (
        f"예상 형태 [3, 224, 224], 실제 형태 {x.shape[1:]}"
    )

def test_data_normalization_bounds(sample_batch):
    """(2) 정규화 범위 확인: 극단적인 이상치(255 등)가 없어야 함"""
    x, _ = sample_batch
    # 느슨한 경계 적용 (-5.0 ~ 5.0)
    assert x.min() >= -5.0 and x.max() <= 5.0, (
        "픽셀 값이 정규화 범위를 벗어났습니다. 전처리 누락을 확인하세요."
    )

def test_label_validity(sample_batch):
    """(3) 레이블 유효성: 결측치 없고 0~21 사이 정수"""
    _, y = sample_batch
    assert not torch.isnan(y.float()).any(), "레이블에 결측치(NaN)가 있습니다."
    assert (y >= 0).all() and (y <= 21).all(), "레이블 값이 0~21 범위를 벗어났습니다."
    assert y.dtype in [torch.int32, torch.int64], "레이블은 정수형이어야 합니다."