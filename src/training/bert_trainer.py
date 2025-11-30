"""BERT训练器模块"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.bert.bert_classifier import BertClassifier
from ..evaluation.metrics import calculate_metrics
from ..utils.file_utils import ensure_dir
from .trainer import BaseTrainer, EarlyStopping, CheckpointManager


class BertTrainer(BaseTrainer):
    """
    BERT训练器
    支持混合精度训练、学习率调度、早停
    """

    def __init__(
        self,
        model: BertClassifier,
        save_dir: str = "models/bert",
        device: str = "cuda",
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 3,
        logging_steps: int = 100,
    ):
        """
        初始化BERT训练器

        Args:
            model: BERT分类器
            save_dir: 保存目录
            device: 设备
            epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_ratio: 预热比例
            max_grad_norm: 梯度裁剪
            fp16: 是否使用混合精度
            gradient_accumulation_steps: 梯度累积步数
            early_stopping_patience: 早停耐心值
            logging_steps: 日志步数
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, min_delta=0.001, mode="max"
        )

        checkpoint_manager = CheckpointManager(
            save_dir=Path(save_dir) / "checkpoints",
            save_best_only=True,
            save_total_limit=3,
            metric_name="f1",
            mode="max",
        )

        super().__init__(model, save_dir, early_stopping, checkpoint_manager)

        self.device = device if torch.cuda.is_available() else "cpu"
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16 and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps

        # 模型移至设备
        self.model = model.to(self.device)

        # 优化器和调度器(延迟初始化)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def _init_optimizer(self, num_training_steps: int) -> None:
        """初始化优化器和调度器"""
        try:
            from transformers import AdamW, get_linear_schedule_with_warmup
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # 优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # 学习率调度器
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 混合精度
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_dataloader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0
        num_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

        for step, batch in enumerate(progress_bar):
            # 移至设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播(混合精度)
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps

            # 反向传播
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_steps += 1

            # 梯度累积
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # 更新参数
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_steps
        return {"loss": avg_loss}

    def validate(
        self, val_dataloader: DataLoader, id2label: Optional[Dict[int, str]] = None
    ) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_dataloader: 验证数据加载器
            id2label: ID到标签的映射

        Returns:
            验证指标
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                total_loss += outputs["loss"].item()

                preds = torch.argmax(outputs["logits"], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(val_dataloader)

        # 转换为标签名
        if id2label:
            all_preds = [id2label.get(p, str(p)) for p in all_preds]
            all_labels = [id2label.get(l, str(l)) for l in all_labels]

        metrics = calculate_metrics(all_labels, all_preds)
        metrics["loss"] = avg_loss

        return metrics

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        id2label: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        完整训练流程

        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            id2label: ID到标签的映射

        Returns:
            训练结果
        """
        # 初始化优化器
        num_training_steps = len(train_dataloader) * self.epochs
        self._init_optimizer(num_training_steps)

        best_metrics = None

        for epoch in range(self.epochs):
            # 训练
            train_metrics = self.train_epoch(train_dataloader, epoch)
            self.train_history.append(train_metrics)

            print(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")

            # 验证
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader, id2label)
                self.val_history.append(val_metrics)

                print(
                    f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}, "
                    f"F1: {val_metrics.get('f1', 0):.4f}"
                )

                # 早停检查
                if self.early_stopping:
                    if self.early_stopping(val_metrics.get("f1", 0)):
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                # 保存检查点
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        state=self.model.state_dict(), metrics=val_metrics, step=epoch
                    )

                if best_metrics is None or val_metrics.get("f1", 0) > best_metrics.get(
                    "f1", 0
                ):
                    best_metrics = val_metrics

        # 保存最终模型
        self._save_model()

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_metrics": best_metrics,
        }

    def _save_model(self) -> None:
        """保存模型"""
        save_path = Path(self.save_dir) / "best_model"
        ensure_dir(save_path)
        self.model.save(save_path)

    def load(self, model_dir: Optional[Union[str, Path]] = None) -> "BertTrainer":
        """加载模型"""
        if model_dir is None:
            model_dir = Path(self.save_dir) / "best_model"
        self.model.load(model_dir)
        return self
