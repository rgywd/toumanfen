"""BERT文本分类器模块"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import BertConfig, get_bert_config


class BertClassifier(nn.Module):
    """
    BERT文本分类器
    基于transformers库
    """

    def __init__(
        self,
        config: Optional[BertConfig] = None,
        num_labels: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        初始化BERT分类器

        Args:
            config: BERT配置对象
            num_labels: 类别数量(会覆盖config中的设置)
            model_name: 模型名称(会覆盖config中的设置)
        """
        super().__init__()

        # 加载配置
        if config is None:
            config = get_bert_config()

        if num_labels is not None:
            config.num_labels = num_labels
        if model_name is not None:
            config.model_name = model_name

        self.config = config

        # 延迟加载transformers
        self._bert = None
        self._classifier = None
        self._dropout = None
        self._initialized = False

    def _init_model(self) -> None:
        """初始化模型(延迟加载)"""
        if self._initialized:
            return

        try:
            from transformers import BertModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        # 加载BERT模型
        self._bert = BertModel.from_pretrained(self.config.model_name)
        hidden_size = self._bert.config.hidden_size

        # 是否冻结BERT参数
        if self.config.freeze_bert:
            for param in self._bert.parameters():
                param.requires_grad = False

        # Dropout层
        self._dropout = nn.Dropout(self.config.classifier_dropout)

        # 分类器
        classifier_input_size = (
            self.config.classifier_hidden_size
            if self.config.classifier_hidden_size
            else hidden_size
        )

        if self.config.classifier_hidden_size:
            self._classifier = nn.Sequential(
                nn.Linear(hidden_size, self.config.classifier_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.classifier_dropout),
                nn.Linear(self.config.classifier_hidden_size, self.config.num_labels),
            )
        else:
            self._classifier = nn.Linear(hidden_size, self.config.num_labels)

        self._initialized = True

    @property
    def bert(self) -> nn.Module:
        """获取BERT模型"""
        self._init_model()
        return self._bert

    @property
    def classifier(self) -> nn.Module:
        """获取分类器"""
        self._init_model()
        return self._classifier

    @property
    def dropout(self) -> nn.Module:
        """获取Dropout层"""
        self._init_model()
        return self._dropout

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入ID张量
            attention_mask: 注意力掩码
            token_type_ids: token类型ID
            labels: 标签(用于计算损失)

        Returns:
            包含logits和loss(如果提供labels)的字典
        """
        self._init_model()

        # BERT编码
        outputs = self._bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 获取[CLS]向量
        if self.config.use_pooler:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # Dropout
        pooled_output = self._dropout(pooled_output)

        # 分类
        logits = self._classifier(pooled_output)

        result = {"logits": logits}

        # 计算损失
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result

    def predict(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测

        Args:
            input_ids: 输入ID张量
            attention_mask: 注意力掩码

        Returns:
            (预测标签, 预测概率)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        保存模型

        Args:
            save_dir: 保存目录
        """
        self._init_model()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存BERT
        self._bert.save_pretrained(save_dir / "bert")

        # 保存分类器
        torch.save(self._classifier.state_dict(), save_dir / "classifier.pt")

        # 保存配置
        import json

        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, load_dir: Union[str, Path]) -> "BertClassifier":
        """
        加载模型

        Args:
            load_dir: 加载目录

        Returns:
            self
        """
        try:
            from transformers import BertModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        load_dir = Path(load_dir)

        # 加载配置
        import json

        with open(load_dir / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        self.config = BertConfig.from_dict(config_dict)

        # 加载BERT
        self._bert = BertModel.from_pretrained(load_dir / "bert")
        hidden_size = self._bert.config.hidden_size

        # 初始化Dropout
        self._dropout = nn.Dropout(self.config.classifier_dropout)

        # 初始化分类器
        if self.config.classifier_hidden_size:
            self._classifier = nn.Sequential(
                nn.Linear(hidden_size, self.config.classifier_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.classifier_dropout),
                nn.Linear(self.config.classifier_hidden_size, self.config.num_labels),
            )
        else:
            self._classifier = nn.Linear(hidden_size, self.config.num_labels)

        # 加载分类器权重
        self._classifier.load_state_dict(torch.load(load_dir / "classifier.pt"))

        self._initialized = True
        return self

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, Path], num_labels: int = 10
    ) -> "BertClassifier":
        """
        从预训练模型加载

        Args:
            model_name_or_path: 模型名称或路径
            num_labels: 类别数量

        Returns:
            BertClassifier实例
        """
        model_path = Path(model_name_or_path)

        if model_path.exists() and (model_path / "config.json").exists():
            # 从保存的目录加载
            model = cls()
            model.load(model_path)
        else:
            # 从HuggingFace加载
            config = get_bert_config(str(model_name_or_path))
            config.num_labels = num_labels
            model = cls(config=config)

        return model
