# src/train.py
import pytorch_lightning as pl
from src.model import FacilityViTClassifier
from src.data import FacilityDataModule

if __name__ == "__main__":
    # ViT 파인튜닝은 BERT보다 살짝 높은 학습률이 좋습니다 (예: 1e-4)
    model = FacilityViTClassifier(lr=1e-4, num_classes=22)
    data = FacilityDataModule(batch_size=32, num_workers=4)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",                  # Day 2: 메모리 반값, 속도 2배 향상
        max_epochs=20,                         # ViT Fine-tuning은 보통 10~20 에포크 권장
        gradient_clip_val=1.0,                 # Day 3: 그래디언트 폭주(NaN) 방지
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                # 기획서 핵심 메트릭: Recall 중심 (또는 val_f1 사용 가능)
                monitor="val_recall", mode="max", 
                save_top_k=1, save_last=True,
                filename="best-{epoch}-{val_recall:.3f}",
            ),
            pl.callbacks.EarlyStopping(
                # Recall이 3번 이상 개선되지 않으면 조기 종료
                monitor="val_recall", patience=3, mode="max" 
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.loggers.WandbLogger(project="gym-facility-safety-inspection"),
    )

    trainer.fit(model, datamodule=data)
    # 최고 성능을 냈던 가중치(best)를 불러와서 최종 검증 수행
    trainer.validate(model, datamodule=data, ckpt_path="best")