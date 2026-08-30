# src/model.py
import torch
import pytorch_lightning as pl
from transformers import AutoModelForImageClassification
from torchmetrics import Recall, F1Score

class FacilityViTClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 22,
        lr: float = 1e-4, # ViT 파인튜닝용 LR
        warmup_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()  # 하이퍼파라미터 자동 저장 및 W&B 로깅

        # Hugging Face에서 사전 학습된 ViT를 로드합니다.
        # 기존 1000개 클래스 헤드를 버리고 22개 클래스로 교체하기 위해 ignore_mismatched_sizes=True 필수
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # 메트릭 — 기획서(Part A)에서 강조한 Recall과 보조 지표 F1 추가
        self.train_recall = Recall(task="multiclass", num_classes=num_labels, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_labels, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def forward(self, pixel_values, labels=None):
        # 텍스트(input_ids) 대신 이미지 픽셀값(pixel_values)을 받습니다.
        return self.model(pixel_values=pixel_values, labels=labels)

    def _shared_step(self, batch):
        """학습/검증 공통 로직 — 수업 예시 패턴 완벽 적용"""
        outputs = self(
            pixel_values=batch['pixel_values'],
            labels=batch['labels']
        )
        preds = outputs.logits.argmax(dim=-1)
        return outputs.loss, preds, batch['labels']

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_recall(preds, labels)
        
        # on_step=True 로 설정하면 학습 과정이 실시간으로 촘촘히 W&B에 찍힙니다
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_recall", self.train_recall, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

    def configure_optimizers(self):
        """OneCycleLR 스케줄러 - 수업 예시 그대로 활용 (성능 최상)"""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}