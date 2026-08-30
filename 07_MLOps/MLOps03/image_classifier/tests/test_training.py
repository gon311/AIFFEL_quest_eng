# tests/test_training.py
import pytorch_lightning as pl
from src.model import FacilityViTClassifier
from src.data import FacilityDataModule

def test_overfit_one_batch():
    """
    1개 배치를 완벽하게 암기할 수 있는지 확인합니다.
    이것이 실패하면 데이터 매칭, 정규화, 혹은 모델 구조에 치명적 버그가 있다는 뜻입니다.
    """
    # 1. 암기 테스트이므로 평소(1e-4)보다 10배 높은 학습률 설정
    model = FacilityViTClassifier(lr=1e-3, num_classes=22) 
    
    # 2. 빠른 디버깅을 위해 num_workers는 0으로, 배치는 작게 설정
    data = FacilityDataModule(data_dir="./data/raw", batch_size=8, num_workers=0)

    # 3. Lightning의 overfit_batches 기능을 사용해 1개 배치를 30번 반복 학습
    # 빠른 테스트를 위해 체크포인트와 로거는 모두 끕니다 (Make it Fast)
    trainer = pl.Trainer(
        overfit_batches=1, 
        max_epochs=30,
        accelerator="auto", 
        enable_checkpointing=False, 
        logger=False,
    )
    trainer.fit(model, datamodule=data)

    # 4. 마지막 Epoch의 Train Loss가 0에 가깝게 떨어졌는지 확인
    # (train_loss_epoch는 model.py에서 on_epoch=True로 로깅했기 때문에 꺼낼 수 있습니다)
    train_loss = trainer.callback_metrics.get("train_loss_epoch", 999)
    assert train_loss < 0.1, f"암기 실패: train_loss={train_loss:.4f} (0.1 이하여야 통과)"