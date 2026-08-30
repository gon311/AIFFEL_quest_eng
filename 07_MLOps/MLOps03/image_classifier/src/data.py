# src/data.py
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
import albumentations as A
import numpy as np

class FacilityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/raw",
        batch_size: int = 32,
        num_workers: int = None,  # None이면 OS에 따라 자동 설정
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── OS별 num_workers 자동 설정 (수업 예시 뼈대 재활용) ──
        if num_workers is not None:
            self._num_workers = num_workers
        elif os.name == 'nt':  # Windows
            self._num_workers = 0   # Windows: 멀티프로세싱 비활성화
        else:  # macOS, Linux
            self._num_workers = min(4, os.cpu_count() or 1)
            
        # ── Vision Augmentation 세팅 ──
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        """데이터 로드 및 이미지 증강(Transform) 함수 매핑"""
        # Hugging Face datasets의 'imagefolder' 기능을 사용하여 로컬 폴더 읽기
        # (폴더 구조: data/raw/train/클래스명/이미지.jpg)
        raw_dataset = load_dataset("imagefolder", data_dir=self.hparams.data_dir)

        def transform_images(examples):
            # albumentations는 numpy array를 받으므로 변환 필요
            images = [np.array(img.convert("RGB")) for img in examples["image"]]
            
            # Train/Val에 따라 다르게 증강 적용
            if stage == "fit" or stage is None:
                transformed = [self.train_transform(image=img)["image"] for img in images]
            else:
                transformed = [self.val_transform(image=img)["image"] for img in images]
                
            # PyTorch 텐서 포맷으로 변경 [C, H, W]
            examples["pixel_values"] = [np.moveaxis(img, -1, 0) for img in transformed]
            return examples

        # 토크나이징 대신 이미지 Transform 함수 매핑 (with_transform 활용)
        raw_dataset = raw_dataset.with_transform(transform_images)

        # HF 모델이 기대하는 키 이름: 'label' → 'labels'로 변경
        if "label" in raw_dataset["train"].column_names:
            raw_dataset = raw_dataset.rename_column("label", "labels")

        self.train_ds = raw_dataset["train"]
        self.val_ds = raw_dataset["validation"]

    def _dataloader_kwargs(self, shuffle=False):
        """OS에 따라 DataLoader 옵션을 자동 조정 (수업 예시 완벽 복사)"""
        kwargs = {
            "batch_size": self.hparams.batch_size,
            "shuffle": shuffle,
            "num_workers": self._num_workers,
        }
        if self._num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["pin_memory"] = True
        return kwargs

    def train_dataloader(self):
        # WeightedRandomSampler가 필요한 경우 shuffle=False 후 여기에 sampler 추가
        return DataLoader(self.train_ds, **self._dataloader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self._dataloader_kwargs(shuffle=False))