"""BERT模型配置模块"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BertConfig:
    """BERT模型配置"""

    # 模型名称
    model_name: str = "hfl/chinese-bert-wwm-ext"

    # 分类配置
    num_labels: int = 10

    # 序列配置
    max_length: int = 128

    # Dropout配置
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1

    # 是否冻结BERT参数
    freeze_bert: bool = False

    # 是否使用池化层
    use_pooler: bool = True

    # 分类器配置
    classifier_hidden_size: Optional[int] = None  # None表示直接使用BERT hidden size

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "classifier_dropout": self.classifier_dropout,
            "freeze_bert": self.freeze_bert,
            "use_pooler": self.use_pooler,
            "classifier_hidden_size": self.classifier_hidden_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BertConfig":
        """从字典创建配置"""
        return cls(**config_dict)


# 预定义配置
BERT_CONFIGS = {
    "chinese-bert-wwm-ext": BertConfig(
        model_name="hfl/chinese-bert-wwm-ext", max_length=128, hidden_dropout_prob=0.1
    ),
    "bert-base-chinese": BertConfig(
        model_name="bert-base-chinese", max_length=128, hidden_dropout_prob=0.1
    ),
    "chinese-roberta-wwm-ext": BertConfig(
        model_name="hfl/chinese-roberta-wwm-ext",
        max_length=128,
        hidden_dropout_prob=0.1,
    ),
    "chinese-macbert-base": BertConfig(
        model_name="hfl/chinese-macbert-base", max_length=128, hidden_dropout_prob=0.1
    ),
}


def get_bert_config(name: str = "chinese-bert-wwm-ext") -> BertConfig:
    """
    获取预定义的BERT配置

    Args:
        name: 配置名称

    Returns:
        BertConfig实例
    """
    if name in BERT_CONFIGS:
        return BERT_CONFIGS[name]
    else:
        # 如果不在预定义列表中，使用默认配置但更新model_name
        config = BertConfig()
        config.model_name = name
        return config
