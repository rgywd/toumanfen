"""数据预处理模块"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import jieba


class BasePreprocessor(ABC):
    """预处理器基类"""

    @abstractmethod
    def process(self, text: str) -> Union[str, List[str], Dict]:
        """处理单条文本"""
        pass

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List:
        """批量处理文本"""
        pass


class BaselinePreprocessor(BasePreprocessor):
    """
    基线模型预处理器
    用于RF+TF-IDF和FastText模型
    功能: 分词、去停用词、去标点、文本清洗
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        min_word_length: int = 1,
        max_word_length: int = 50,
        lower: bool = True,
    ):
        """
        初始化预处理器

        Args:
            stopwords: 停用词集合
            remove_stopwords: 是否去除停用词
            remove_punctuation: 是否去除标点
            min_word_length: 最小词长
            max_word_length: 最大词长
            lower: 是否转小写
        """
        self.stopwords = stopwords or set()
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.lower = lower

        # 标点符号正则
        self.punct_pattern = re.compile(r"[^\w\s\u4e00-\u9fff]")
        # 空白符正则
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """
        清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if self.lower:
            text = text.lower()

        if self.remove_punctuation:
            text = self.punct_pattern.sub(" ", text)

        # 去除多余空白
        text = self.whitespace_pattern.sub(" ", text)
        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        分词

        Args:
            text: 文本

        Returns:
            词列表
        """
        words = list(jieba.cut(text))
        return words

    def filter_words(self, words: List[str]) -> List[str]:
        """
        过滤词语

        Args:
            words: 词列表

        Returns:
            过滤后的词列表
        """
        filtered = []

        for word in words:
            word = word.strip()

            # 长度过滤
            if len(word) < self.min_word_length:
                continue
            if len(word) > self.max_word_length:
                continue

            # 停用词过滤
            if self.remove_stopwords and word in self.stopwords:
                continue

            filtered.append(word)

        return filtered

    def process(self, text: str) -> str:
        """
        处理单条文本

        Args:
            text: 原始文本

        Returns:
            处理后的文本(空格分隔的词)
        """
        text = self.clean_text(text)
        words = self.tokenize(text)
        words = self.filter_words(words)
        return " ".join(words)

    def process_batch(self, texts: List[str]) -> List[str]:
        """
        批量处理文本

        Args:
            texts: 文本列表

        Returns:
            处理后的文本列表
        """
        return [self.process(text) for text in texts]


class BertPreprocessor(BasePreprocessor):
    """
    BERT模型预处理器
    功能: 基本清洗(不去停用词)、Tokenization
    """

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: Optional[str] = None,
    ):
        """
        初始化BERT预处理器

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            truncation: 是否截断
            padding: 填充策略
            return_tensors: 返回张量类型 ("pt" 或 None)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors

        # 清洗正则
        self.whitespace_pattern = re.compile(r"\s+")

    def set_tokenizer(self, tokenizer) -> None:
        """设置tokenizer"""
        self.tokenizer = tokenizer

    def clean_text(self, text: str) -> str:
        """
        基本清洗(保留停用词)

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        # 去除多余空白
        text = self.whitespace_pattern.sub(" ", text)
        text = text.strip()
        return text

    def process(self, text: str) -> Dict:
        """
        处理单条文本

        Args:
            text: 原始文本

        Returns:
            BERT输入字典 (input_ids, attention_mask, token_type_ids)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

        text = self.clean_text(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )

        return encoding

    def process_batch(self, texts: List[str]) -> Dict:
        """
        批量处理文本

        Args:
            texts: 文本列表

        Returns:
            BERT输入字典
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")

        texts = [self.clean_text(text) for text in texts]

        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )

        return encoding
