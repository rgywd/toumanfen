"""基线模型训练器模块"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import cross_val_score

from ..features.tfidf_extractor import TfidfExtractor
from ..models.baseline.random_forest import RFClassifier
from ..models.baseline.fasttext_model import FastTextClassifier
from ..evaluation.metrics import calculate_metrics
from ..utils.file_utils import ensure_dir
from .trainer import BaseTrainer


class BaselineTrainer(BaseTrainer):
    """
    基线模型训练器
    支持RF+TF-IDF和FastText
    """

    def __init__(
        self, model_type: str = "rf", save_dir: str = "models/baseline", **model_kwargs
    ):
        """
        初始化基线训练器

        Args:
            model_type: 模型类型 ("rf" 或 "fasttext")
            save_dir: 保存目录
            **model_kwargs: 模型参数
        """
        self.model_type = model_type
        self.save_dir = Path(save_dir)
        self.model_kwargs = model_kwargs

        # 创建模型
        if model_type == "rf":
            self.model = RFClassifier(**model_kwargs)
            self.tfidf = TfidfExtractor()
        elif model_type == "fasttext":
            self.model = FastTextClassifier(**model_kwargs)
            self.tfidf = None
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        super().__init__(self.model, save_dir)

    def train_epoch(self, *args, **kwargs) -> Dict[str, float]:
        """基线模型不需要分epoch训练"""
        pass

    def validate(self, X_val, y_val: List[str]) -> Dict[str, float]:
        """
        验证模型

        Args:
            X_val: 验证集特征/文本
            y_val: 验证集标签

        Returns:
            评估指标
        """
        if self.model_type == "rf":
            y_pred = self.model.predict(X_val)
        else:
            y_pred, _ = self.model.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred)
        return metrics

    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        cross_validation: int = 0,
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本
            val_labels: 验证标签
            cross_validation: 交叉验证折数(0表示不使用)

        Returns:
            训练结果
        """
        result = {}

        if self.model_type == "rf":
            result = self._train_rf(
                train_texts, train_labels, val_texts, val_labels, cross_validation
            )
        else:
            result = self._train_fasttext(
                train_texts, train_labels, val_texts, val_labels
            )

        return result

    def _train_rf(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]],
        val_labels: Optional[List[str]],
        cross_validation: int,
    ) -> Dict[str, Any]:
        """训练随机森林"""
        # TF-IDF特征提取
        X_train = self.tfidf.fit_transform(train_texts)

        # 标签编码
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_labels)

        result = {"model_type": "rf"}

        # 交叉验证
        if cross_validation > 0:
            cv_scores = cross_val_score(
                self.model.model,
                X_train,
                y_train,
                cv=cross_validation,
                scoring="f1_macro",
            )
            result["cv_scores"] = cv_scores.tolist()
            result["cv_mean"] = cv_scores.mean()
            result["cv_std"] = cv_scores.std()

        # 训练
        self.model.train(X_train, y_train)

        # 训练集评估
        train_pred = self.model.predict(X_train)
        train_pred_labels = self.label_encoder.inverse_transform(train_pred)
        train_metrics = calculate_metrics(train_labels, train_pred_labels)
        result["train_metrics"] = train_metrics

        # 验证集评估
        if val_texts is not None and val_labels is not None:
            X_val = self.tfidf.transform(val_texts)
            val_pred = self.model.predict(X_val)
            val_pred_labels = self.label_encoder.inverse_transform(val_pred)
            val_metrics = calculate_metrics(val_labels, val_pred_labels)
            result["val_metrics"] = val_metrics

        # 保存
        self._save_rf()

        return result

    def _train_fasttext(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]],
        val_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        """训练FastText"""
        # 训练
        save_path = self.save_dir / "fasttext" / "model.bin"
        self.model.train(train_texts, train_labels, save_path)

        result = {"model_type": "fasttext"}

        # 训练集评估
        train_pred, _ = self.model.predict(train_texts)
        train_metrics = calculate_metrics(train_labels, train_pred)
        result["train_metrics"] = train_metrics

        # 验证集评估
        if val_texts is not None and val_labels is not None:
            val_pred, _ = self.model.predict(val_texts)
            val_metrics = calculate_metrics(val_labels, val_pred)
            result["val_metrics"] = val_metrics

        return result

    def _save_rf(self) -> None:
        """保存RF模型和TF-IDF"""
        save_dir = self.save_dir / "rf_tfidf"
        ensure_dir(save_dir)

        self.model.save(save_dir / "model.pkl")
        self.tfidf.save(save_dir / "vectorizer.pkl")

        # 保存标签编码器
        from ..utils.file_utils import save_pickle

        save_pickle(self.label_encoder, save_dir / "label_encoder.pkl")

    def load(self, model_dir: Optional[Union[str, Path]] = None) -> "BaselineTrainer":
        """
        加载模型

        Args:
            model_dir: 模型目录

        Returns:
            self
        """
        if model_dir is None:
            model_dir = self.save_dir
        model_dir = Path(model_dir)

        if self.model_type == "rf":
            rf_dir = model_dir / "rf_tfidf"
            self.model.load(rf_dir / "model.pkl")
            self.tfidf.load(rf_dir / "vectorizer.pkl")

            from ..utils.file_utils import load_pickle

            self.label_encoder = load_pickle(rf_dir / "label_encoder.pkl")
        else:
            ft_path = model_dir / "fasttext" / "model.bin"
            self.model.load(ft_path)

        return self
