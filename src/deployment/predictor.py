"""通用预测器模块"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data.preprocessor import BaselinePreprocessor, BertPreprocessor
from .model_loader import UniversalModelLoader


class UniversalPredictor:
    """
    通用预测器
    支持所有模型类型的预测
    """

    def __init__(
        self,
        models_dir: str = "models",
        default_model: str = "bert",
        device: str = "cuda",
    ):
        """
        初始化预测器

        Args:
            models_dir: 模型目录
            default_model: 默认模型类型
            device: 设备
        """
        self.model_loader = UniversalModelLoader(models_dir)
        self.default_model = default_model
        self.device = device

        # 预处理器
        self.baseline_preprocessor = BaselinePreprocessor()
        self.bert_preprocessor = BertPreprocessor()

        # 缓存已加载的模型
        self._models: Dict[str, Dict] = {}

    def _ensure_model_loaded(self, model_type: str) -> Dict[str, Any]:
        """确保模型已加载"""
        if model_type not in self._models:
            self._models[model_type] = self.model_loader.load_model(model_type)

            # 设置BERT tokenizer
            if model_type in ["bert", "quantized", "pruned", "distilled"]:
                tokenizer = self._models[model_type].get("tokenizer")
                if tokenizer:
                    self.bert_preprocessor.set_tokenizer(tokenizer)

        return self._models[model_type]

    def predict(
        self,
        text: Union[str, List[str]],
        model_type: Optional[str] = None,
        return_proba: bool = False,
    ) -> Union[str, List[str], Tuple[Union[str, List[str]], np.ndarray]]:
        """
        预测单条或多条文本

        Args:
            text: 文本或文本列表
            model_type: 模型类型
            return_proba: 是否返回概率

        Returns:
            预测类别(和概率)
        """
        if model_type is None:
            model_type = self.default_model

        model_type = model_type.lower()

        # 加载模型
        model_data = self._ensure_model_loaded(model_type)

        # 处理输入
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # 预测
        if model_type == "rf":
            predictions, probas = self._predict_rf(texts, model_data)
        elif model_type == "fasttext":
            predictions, probas = self._predict_fasttext(texts, model_data)
        else:  # bert and optimized models
            predictions, probas = self._predict_bert(texts, model_data)

        # 处理输出
        if is_single:
            predictions = predictions[0]
            probas = probas[0] if probas is not None else None

        if return_proba:
            return predictions, probas
        return predictions

    def _predict_rf(
        self, texts: List[str], model_data: Dict
    ) -> Tuple[List[str], np.ndarray]:
        """RF模型预测"""
        model = model_data["model"]
        tfidf = model_data["tfidf"]
        label_encoder = model_data["label_encoder"]

        # 预处理
        processed = self.baseline_preprocessor.process_batch(texts)

        # 特征提取
        features = tfidf.transform(processed)

        # 预测
        pred_ids = model.predict(features)
        probas = model.predict_proba(features)

        # 转换标签
        predictions = label_encoder.inverse_transform(pred_ids).tolist()

        return predictions, probas

    def _predict_fasttext(
        self, texts: List[str], model_data: Dict
    ) -> Tuple[List[str], np.ndarray]:
        """FastText模型预测"""
        model = model_data["model"]

        # 预处理
        processed = self.baseline_preprocessor.process_batch(texts)

        # 预测
        predictions, probas = model.predict(processed)

        return predictions, probas

    def _predict_bert(
        self, texts: List[str], model_data: Dict
    ) -> Tuple[List[str], np.ndarray]:
        """BERT模型预测"""
        import torch

        model = model_data["model"]
        tokenizer = model_data.get("tokenizer")

        if tokenizer is None:
            raise ValueError("Tokenizer not available for BERT model")

        # 预处理
        self.bert_preprocessor.set_tokenizer(tokenizer)
        encoding = self.bert_preprocessor.process_batch(texts)

        # 转换为tensor
        input_ids = torch.tensor(encoding["input_ids"])
        attention_mask = torch.tensor(encoding["attention_mask"])

        # 移至设备
        device = self.device if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 预测
        preds, probas = model.predict(input_ids, attention_mask)

        preds = preds.cpu().numpy()
        probas = probas.cpu().numpy()

        # 转换为标签名(需要id2label映射)
        predictions = [str(p) for p in preds]

        return predictions, probas

    def predict_with_confidence(
        self, text: Union[str, List[str]], model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        预测并返回置信度

        Args:
            text: 文本
            model_type: 模型类型

        Returns:
            [{"label": ..., "confidence": ...}, ...]
        """
        predictions, probas = self.predict(text, model_type, return_proba=True)

        if isinstance(predictions, str):
            predictions = [predictions]
            probas = [probas]

        results = []
        for pred, proba in zip(predictions, probas):
            if proba is not None:
                confidence = float(np.max(proba))
            else:
                confidence = 1.0
            results.append({"label": pred, "confidence": confidence})

        return results

    def batch_predict(
        self, texts: List[str], model_type: Optional[str] = None, batch_size: int = 32
    ) -> List[str]:
        """
        批量预测

        Args:
            texts: 文本列表
            model_type: 模型类型
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            predictions = self.predict(batch, model_type)
            all_predictions.extend(predictions)

        return all_predictions

    def get_available_models(self) -> Dict[str, bool]:
        """获取可用的模型"""
        return self.model_loader.get_available_models()
