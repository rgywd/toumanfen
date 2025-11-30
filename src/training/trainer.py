"""通用训练器模块"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "max"):
        """
        初始化早停

        Args:
            patience: 容忍的无改善轮数
            min_delta: 最小改善阈值
            mode: "max"表示指标越大越好, "min"表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前评估指标

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class CheckpointManager:
    """检查点管理器"""

    def __init__(
        self,
        save_dir: Union[str, Path],
        save_best_only: bool = True,
        save_total_limit: int = 3,
        metric_name: str = "f1",
        mode: str = "max",
    ):
        """
        初始化检查点管理器

        Args:
            save_dir: 保存目录
            save_best_only: 是否只保存最佳模型
            save_total_limit: 最多保存的检查点数量
            metric_name: 评估指标名称
            mode: "max"或"min"
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.save_total_limit = save_total_limit
        self.metric_name = metric_name
        self.mode = mode

        self.best_score = None
        self.checkpoints: List[Tuple[Path, float]] = []

    def should_save(self, metrics: Dict[str, float]) -> bool:
        """判断是否应该保存"""
        if not self.save_best_only:
            return True

        score = metrics.get(self.metric_name, 0)

        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            return True

        return False

    def save_checkpoint(
        self, state: Dict[str, Any], metrics: Dict[str, float], step: int
    ) -> Optional[Path]:
        """
        保存检查点

        Args:
            state: 模型状态
            metrics: 评估指标
            step: 当前步数

        Returns:
            保存路径(如果保存了)
        """
        import torch

        if not self.should_save(metrics):
            return None

        score = metrics.get(self.metric_name, 0)
        checkpoint_path = self.save_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 保存状态
        torch.save(state, checkpoint_path / "model.pt")

        # 更新检查点列表
        self.checkpoints.append((checkpoint_path, score))

        # 限制检查点数量
        if len(self.checkpoints) > self.save_total_limit:
            # 按分数排序,删除最差的
            if self.mode == "max":
                self.checkpoints.sort(key=lambda x: x[1], reverse=True)
            else:
                self.checkpoints.sort(key=lambda x: x[1])

            # 删除超出的检查点
            while len(self.checkpoints) > self.save_total_limit:
                old_path, _ = self.checkpoints.pop()
                if old_path.exists():
                    import shutil

                    shutil.rmtree(old_path)

        return checkpoint_path


class BaseTrainer(ABC):
    """训练器基类"""

    def __init__(
        self,
        model: Any,
        save_dir: str = "models",
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """
        初始化训练器

        Args:
            model: 模型实例
            save_dir: 模型保存目录
            early_stopping: 早停机制
            checkpoint_manager: 检查点管理器
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager

        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

    @abstractmethod
    def train_epoch(self, *args, **kwargs) -> Dict[str, float]:
        """训练一个epoch"""
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> Dict[str, float]:
        """验证"""
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """完整训练流程"""
        pass

    def get_history(self) -> Dict[str, List]:
        """获取训练历史"""
        return {"train": self.train_history, "val": self.val_history}
