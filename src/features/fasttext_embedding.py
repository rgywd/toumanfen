"""FastText词向量模块"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class FastTextEmbedding:
    """FastText词向量"""

    def __init__(
        self,
        dim: int = 100,
        min_count: int = 2,
        window: int = 5,
        epochs: int = 5,
        model_type: str = "skipgram",
    ):
        """
        初始化FastText词向量

        Args:
            dim: 向量维度
            min_count: 最小词频
            window: 上下文窗口大小
            epochs: 训练轮数
            model_type: 模型类型 ("skipgram" 或 "cbow")
        """
        self.dim = dim
        self.min_count = min_count
        self.window = window
        self.epochs = epochs
        self.model_type = model_type
        self.model = None

    def train_embeddings(
        self, texts: List[str], save_path: Optional[Union[str, Path]] = None
    ) -> "FastTextEmbedding":
        """
        训练FastText词向量

        Args:
            texts: 预处理后的文本列表(空格分隔的词)
            save_path: 模型保存路径

        Returns:
            self
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("Please install fasttext: pip install fasttext")

        # 准备训练数据(写入临时文件)
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for text in texts:
                f.write(text + "\n")
            temp_path = f.name

        # 训练模型
        model_method = "skipgram" if self.model_type == "skipgram" else "cbow"

        self.model = fasttext.train_unsupervised(
            temp_path,
            model=model_method,
            dim=self.dim,
            minCount=self.min_count,
            ws=self.window,
            epoch=self.epochs,
        )

        # 删除临时文件
        Path(temp_path).unlink()

        # 保存模型
        if save_path:
            self.save(save_path)

        return self

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        获取词向量

        Args:
            word: 词

        Returns:
            词向量
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.get_word_vector(word)

    def get_sentence_vector(self, text: str) -> np.ndarray:
        """
        获取句子向量(词向量平均)

        Args:
            text: 预处理后的文本(空格分隔的词)

        Returns:
            句子向量
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.get_sentence_vector(text)

    def get_batch_vectors(self, texts: List[str]) -> np.ndarray:
        """
        批量获取句子向量

        Args:
            texts: 预处理后的文本列表

        Returns:
            向量矩阵 (n_samples, dim)
        """
        vectors = [self.get_sentence_vector(text) for text in texts]
        return np.array(vectors)

    def save(self, save_path: Union[str, Path]) -> None:
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.save_model(str(save_path))

    def load(self, load_path: Union[str, Path]) -> "FastTextEmbedding":
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
        self.dim = self.model.get_dimension()
        return self

    @property
    def vocabulary(self) -> List[str]:
        """获取词汇表"""
        if self.model is None:
            return []
        return self.model.get_words()
