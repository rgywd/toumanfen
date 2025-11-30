"""FastText分类器模块"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class FastTextClassifier:
    """
    FastText文本分类器
    基于fasttext库
    """

    def __init__(
        self,
        dim: int = 100,
        epoch: int = 25,
        lr: float = 0.5,
        wordNgrams: int = 2,
        minCount: int = 2,
        loss: str = "softmax",
        bucket: int = 2000000,
        **kwargs,
    ):
        """
        初始化FastText分类器

        Args:
            dim: 词向量维度
            epoch: 训练轮数
            lr: 学习率
            wordNgrams: n-gram大小
            minCount: 最小词频
            loss: 损失函数 ("softmax", "hs", "ova")
            bucket: hash bucket数量
            **kwargs: 其他fasttext参数
        """
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.minCount = minCount
        self.loss = loss
        self.bucket = bucket
        self.kwargs = kwargs
        self.model = None

    def train(
        self,
        texts: List[str],
        labels: List[str],
        save_path: Optional[Union[str, Path]] = None,
    ) -> "FastTextClassifier":
        """
        训练FastText分类模型

        Args:
            texts: 预处理后的文本列表
            labels: 标签列表
            save_path: 模型保存路径

        Returns:
            self
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("Please install fasttext: pip install fasttext")

        # 准备训练数据(fasttext格式: __label__xxx text)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for text, label in zip(texts, labels):
                f.write(f"__label__{label} {text}\n")
            temp_path = f.name

        # 训练模型
        self.model = fasttext.train_supervised(
            temp_path,
            dim=self.dim,
            epoch=self.epoch,
            lr=self.lr,
            wordNgrams=self.wordNgrams,
            minCount=self.minCount,
            loss=self.loss,
            bucket=self.bucket,
            **self.kwargs,
        )

        # 删除临时文件
        Path(temp_path).unlink()

        # 保存模型
        if save_path:
            self.save(save_path)

        return self

    def predict(
        self, texts: Union[str, List[str]], k: int = 1
    ) -> Tuple[List[str], np.ndarray]:
        """
        预测类别

        Args:
            texts: 文本或文本列表
            k: 返回top-k预测

        Returns:
            (预测标签列表, 置信度数组)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        probabilities = []

        for text in texts:
            labels, probs = self.model.predict(text, k=k)
            # 去除__label__前缀
            labels = [label.replace("__label__", "") for label in labels]
            predictions.append(labels[0] if k == 1 else labels)
            probabilities.append(probs[0] if k == 1 else probs)

        return predictions, np.array(probabilities)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        预测所有类别的概率

        Args:
            texts: 文本列表

        Returns:
            概率矩阵 (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        labels = self.get_labels()
        n_classes = len(labels)

        proba_matrix = np.zeros((len(texts), n_classes))

        for i, text in enumerate(texts):
            pred_labels, probs = self.model.predict(text, k=n_classes)
            for label, prob in zip(pred_labels, probs):
                label = label.replace("__label__", "")
                if label in labels:
                    idx = labels.index(label)
                    proba_matrix[i, idx] = prob

        return proba_matrix

    def get_labels(self) -> List[str]:
        """获取所有类别标签"""
        if self.model is None:
            return []
        labels = self.model.get_labels()
        return [label.replace("__label__", "") for label in labels]

    def save(self, save_path: Union[str, Path]) -> None:
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.save_model(str(save_path))

    def load(self, load_path: Union[str, Path]) -> "FastTextClassifier":
        """
        加载模型

        Args:
            load_path: 加载路径

        Returns:
            self
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("Please install fasttext: pip install fasttext")

        self.model = fasttext.load_model(str(load_path))
        return self
