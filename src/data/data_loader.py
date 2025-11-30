"""数据加载器模块"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..utils.file_utils import read_lines


class DataLoader:
    """数据加载器"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        初始化数据加载器

        Args:
            data_dir: 原始数据目录
        """
        self.data_dir = Path(data_dir)
        self._stopwords: Optional[set] = None
        self._class_labels: Optional[List[str]] = None

    def load_train(self) -> Tuple[List[str], List[str]]:
        """加载训练集"""
        return self._load_data("train.txt")

    def load_test(self) -> Tuple[List[str], List[str]]:
        """加载测试集"""
        return self._load_data("test.txt")

    def load_dev(self) -> Tuple[List[str], List[str]]:
        """加载验证集"""
        return self._load_data("dev.txt")

    def _load_data(self, filename: str) -> Tuple[List[str], List[str]]:
        """
        加载数据文件

        Args:
            filename: 文件名

        Returns:
            (文本列表, 标签列表)
        """
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        texts, labels = [], []
        lines = read_lines(file_path)

        for line in lines:
            text, label = parse_line(line)
            if text and label:
                texts.append(text)
                labels.append(label)

        return texts, labels

    def load_stopwords(self) -> set:
        """加载停用词表"""
        if self._stopwords is not None:
            return self._stopwords

        stopwords_path = self.data_dir / "stopwords.txt"

        if stopwords_path.exists():
            self._stopwords = set(read_lines(stopwords_path))
        else:
            self._stopwords = set()

        return self._stopwords

    def load_class_labels(self) -> List[str]:
        """加载类别标签"""
        if self._class_labels is not None:
            return self._class_labels

        class_path = self.data_dir / "class.txt"

        if class_path.exists():
            self._class_labels = read_lines(class_path)
        else:
            self._class_labels = []

        return self._class_labels


def parse_line(line: str, sep: str = "\t") -> Tuple[str, str]:
    """
    解析单行数据

    Args:
        line: 数据行
        sep: 分隔符

    Returns:
        (文本, 标签)
    """
    parts = line.strip().split(sep)

    if len(parts) >= 2:
        text = parts[0].strip()
        label = parts[1].strip()
        return text, label

    return "", ""


def load_raw_data(
    file_path: Union[str, Path], sep: str = "\t"
) -> Tuple[List[str], List[str]]:
    """
    加载原始数据文件

    Args:
        file_path: 文件路径
        sep: 分隔符

    Returns:
        (文本列表, 标签列表)
    """
    texts, labels = [], []
    lines = read_lines(file_path)

    for line in lines:
        text, label = parse_line(line, sep)
        if text and label:
            texts.append(text)
            labels.append(label)

    return texts, labels


def load_stopwords(file_path: Union[str, Path]) -> set:
    """
    加载停用词表

    Args:
        file_path: 停用词文件路径

    Returns:
        停用词集合
    """
    if not Path(file_path).exists():
        return set()
    return set(read_lines(file_path))


def load_class_labels(file_path: Union[str, Path]) -> List[str]:
    """
    加载类别标签

    Args:
        file_path: 类别文件路径

    Returns:
        类别列表
    """
    if not Path(file_path).exists():
        return []
    return read_lines(file_path)
