"""TF-IDF特征提取模块"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.file_utils import load_pickle, save_pickle


class TfidfExtractor:
    """TF-IDF特征提取器"""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
        **kwargs
    ):
        """
        初始化TF-IDF提取器

        Args:
            max_features: 最大特征数
            ngram_range: n-gram范围
            min_df: 最小文档频率
            max_df: 最大文档频率
            sublinear_tf: 是否使用对数TF
            **kwargs: 其他TfidfVectorizer参数
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            **kwargs
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "TfidfExtractor":
        """
        在训练集上拟合TF-IDF

        Args:
            texts: 预处理后的文本列表

        Returns:
            self
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self

    def transform(self, texts: List[str]) -> spmatrix:
        """
        转换文本为TF-IDF向量

        Args:
            texts: 预处理后的文本列表

        Returns:
            TF-IDF稀疏矩阵
        """
        if not self._is_fitted:
            raise RuntimeError("TfidfExtractor must be fitted before transform")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> spmatrix:
        """
        拟合并转换

        Args:
            texts: 预处理后的文本列表

        Returns:
            TF-IDF稀疏矩阵
        """
        self._is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        """获取特征名列表"""
        return self.vectorizer.get_feature_names_out().tolist()

    def get_vocabulary(self) -> dict:
        """获取词汇表"""
        return self.vectorizer.vocabulary_

    def save(self, save_path: Union[str, Path]) -> None:
        """
        保存vectorizer

        Args:
            save_path: 保存路径
        """
        save_pickle(self.vectorizer, save_path)

    def load(self, load_path: Union[str, Path]) -> "TfidfExtractor":
        """
        加载vectorizer

        Args:
            load_path: 加载路径

        Returns:
            self
        """
        self.vectorizer = load_pickle(load_path)
        self._is_fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        """是否已拟合"""
        return self._is_fitted
