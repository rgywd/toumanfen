"""词表构建模块"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..utils.file_utils import ensure_dir, read_lines, write_lines


class VocabularyBuilder:
    """词表构建器"""

    # 特殊标记
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"

    def __init__(
        self,
        min_freq: int = 2,
        max_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        初始化词表构建器

        Args:
            min_freq: 最小词频
            max_size: 词表最大大小
            special_tokens: 特殊标记列表
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.special_tokens = special_tokens or [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
        ]

        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

    def build(self, texts: List[str]) -> "VocabularyBuilder":
        """
        从文本构建词表

        Args:
            texts: 分词后的文本列表(空格分隔)

        Returns:
            self
        """
        # 统计词频
        for text in texts:
            words = text.split()
            self.word_freq.update(words)

        # 添加特殊标记
        for idx, token in enumerate(self.special_tokens):
            self.word2id[token] = idx
            self.id2word[idx] = token

        # 按频率排序添加词
        idx = len(self.special_tokens)
        for word, freq in self.word_freq.most_common():
            if freq < self.min_freq:
                break
            if self.max_size and idx >= self.max_size:
                break
            if word not in self.word2id:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1

        return self

    def __len__(self) -> int:
        return len(self.word2id)

    def encode(self, text: str) -> List[int]:
        """
        编码文本为ID序列

        Args:
            text: 分词后的文本

        Returns:
            ID列表
        """
        words = text.split()
        unk_id = self.word2id.get(self.UNK_TOKEN, 1)
        return [self.word2id.get(word, unk_id) for word in words]

    def decode(self, ids: List[int]) -> str:
        """
        解码ID序列为文本

        Args:
            ids: ID列表

        Returns:
            文本
        """
        words = [self.id2word.get(i, self.UNK_TOKEN) for i in ids]
        return " ".join(words)

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        保存词表

        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        ensure_dir(save_dir)

        # 保存vocab.txt
        vocab_path = save_dir / "vocab.txt"
        vocab_list = [self.id2word[i] for i in range(len(self.id2word))]
        write_lines(vocab_list, vocab_path)

        # 保存word2id.json
        word2id_path = save_dir / "word2id.json"
        with open(word2id_path, "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: Union[str, Path]) -> "VocabularyBuilder":
        """
        加载词表

        Args:
            save_dir: 保存目录

        Returns:
            self
        """
        save_dir = Path(save_dir)

        # 加载vocab.txt
        vocab_path = save_dir / "vocab.txt"
        if vocab_path.exists():
            vocab_list = read_lines(vocab_path)
            self.word2id = {word: idx for idx, word in enumerate(vocab_list)}
            self.id2word = {idx: word for idx, word in enumerate(vocab_list)}

        return self


def build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建标签映射

    Args:
        labels: 标签列表

    Returns:
        (label2id, id2label)
    """
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    return label2id, id2label


def save_label_mapping(label2id: Dict[str, int], save_path: Union[str, Path]) -> None:
    """
    保存标签映射

    Args:
        label2id: 标签到ID的映射
        save_path: 保存路径
    """
    ensure_dir(Path(save_path).parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


def load_label_mapping(load_path: Union[str, Path]) -> Dict[str, int]:
    """
    加载标签映射

    Args:
        load_path: 加载路径

    Returns:
        label2id映射
    """
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)
