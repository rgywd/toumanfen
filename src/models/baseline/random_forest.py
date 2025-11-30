"""随机森林分类器模块"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier

from ...utils.file_utils import load_pickle, save_pickle


class RFClassifier:
    """
    随机森林分类器
    结合TF-IDF特征使用
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 50,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        """
        初始化随机森林分类器

        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            min_samples_split: 内部节点再划分所需最小样本数
            min_samples_leaf: 叶子节点最少样本数
            n_jobs: 并行任务数
            random_state: 随机种子
            **kwargs: 其他RandomForestClassifier参数
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        self._is_fitted = False

    def train(self, X: Union[np.ndarray, spmatrix], y: np.ndarray) -> "RFClassifier":
        """
        训练随机森林

        Args:
            X: 特征矩阵
            y: 标签数组

        Returns:
            self
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, spmatrix]) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵

        Returns:
            预测标签数组
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, spmatrix]) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            预测概率矩阵
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if not self._is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_

    def save(self, save_path: Union[str, Path]) -> None:
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        save_pickle(self.model, save_path)

    def load(self, load_path: Union[str, Path]) -> "RFClassifier":
        """
        加载模型

        Args:
            load_path: 加载路径

        Returns:
            self
        """
        self.model = load_pickle(load_path)
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted

    @property
    def classes(self) -> np.ndarray:
        """获取类别列表"""
        if not self._is_fitted:
            return np.array([])
        return self.model.classes_
