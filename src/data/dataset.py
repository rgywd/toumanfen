"""PyTorch Dataset模块"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    BERT模型的PyTorch Dataset
    支持动态batch padding
    """

    def __init__(self, encodings: Dict, labels: Optional[List[int]] = None):
        """
        初始化Dataset

        Args:
            encodings: tokenizer编码结果
            labels: 标签列表
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item


class BaselineDataset:
    """
    基线模型的数据集类
    用于RF和FastText模型
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        label2id: Optional[Dict[str, int]] = None,
    ):
        """
        初始化Dataset

        Args:
            texts: 处理后的文本列表(空格分隔的词)
            labels: 标签列表
            label2id: 标签到ID的映射
        """
        self.texts = texts
        self.labels = labels
        self.label2id = label2id or {}

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Union[str, Tuple[str, str]]:
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx]

    def get_texts(self) -> List[str]:
        """获取所有文本"""
        return self.texts

    def get_labels(self) -> Optional[List[str]]:
        """获取所有标签"""
        return self.labels

    def get_label_ids(self) -> Optional[List[int]]:
        """获取标签ID列表"""
        if self.labels is None or not self.label2id:
            return None
        return [self.label2id.get(label, 0) for label in self.labels]


class DataCollator:
    """
    数据整理器
    用于动态padding
    """

    def __init__(self, tokenizer, max_length: int = 128):
        """
        初始化数据整理器

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一个batch的数据

        Args:
            batch: 样本列表

        Returns:
            整理后的batch字典
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # 动态padding到batch内最大长度
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_len - len(ids)

            if padding_length > 0:
                ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
            else:
                ids = ids[:max_len]
                mask = mask[:max_len]

            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }

        if "labels" in batch[0]:
            labels = torch.tensor([item["labels"] for item in batch])
            result["labels"] = labels

        return result
