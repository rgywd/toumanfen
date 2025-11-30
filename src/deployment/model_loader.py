"""通用模型加载器模块"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..models.baseline.random_forest import RFClassifier
from ..models.baseline.fasttext_model import FastTextClassifier
from ..models.bert.bert_classifier import BertClassifier
from ..features.tfidf_extractor import TfidfExtractor
from ..utils.file_utils import load_pickle


class UniversalModelLoader:
    """
    通用模型加载器
    支持加载所有类型的模型
    """

    # 支持的模型类型
    SUPPORTED_MODELS = ["rf", "fasttext", "bert", "quantized", "pruned", "distilled"]

    def __init__(self, models_dir: str = "models"):
        """
        初始化模型加载器

        Args:
            models_dir: 模型目录
        """
        self.models_dir = Path(models_dir)
        self._loaded_models: Dict[str, Any] = {}

    def load_model(
        self, model_type: str, model_path: Optional[Union[str, Path]] = None
    ) -> Any:
        """
        加载模型

        Args:
            model_type: 模型类型 ("rf", "fasttext", "bert", "quantized", "pruned", "distilled")
            model_path: 模型路径(可选,默认使用标准路径)

        Returns:
            加载的模型
        """
        model_type = model_type.lower()

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        # 检查是否已加载
        cache_key = f"{model_type}_{model_path}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        # 加载模型
        if model_type == "rf":
            model = self._load_rf(model_path)
        elif model_type == "fasttext":
            model = self._load_fasttext(model_path)
        elif model_type == "bert":
            model = self._load_bert(model_path)
        elif model_type in ["quantized", "pruned", "distilled"]:
            model = self._load_optimized(model_type, model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._loaded_models[cache_key] = model
        return model

    def _load_rf(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载随机森林模型"""
        if model_path is None:
            model_path = self.models_dir / "baseline" / "rf_tfidf"
        else:
            model_path = Path(model_path)

        # 加载模型
        model = RFClassifier()
        model.load(model_path / "model.pkl")

        # 加载TF-IDF
        tfidf = TfidfExtractor()
        tfidf.load(model_path / "vectorizer.pkl")

        # 加载标签编码器
        label_encoder = load_pickle(model_path / "label_encoder.pkl")

        return {
            "model": model,
            "tfidf": tfidf,
            "label_encoder": label_encoder,
            "type": "rf",
        }

    def _load_fasttext(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载FastText模型"""
        if model_path is None:
            model_path = self.models_dir / "baseline" / "fasttext" / "model.bin"
        else:
            model_path = Path(model_path)

        model = FastTextClassifier()
        model.load(model_path)

        return {"model": model, "type": "fasttext"}

    def _load_bert(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载BERT模型"""
        if model_path is None:
            model_path = self.models_dir / "bert" / "best_model"
        else:
            model_path = Path(model_path)

        model = BertClassifier.from_pretrained(model_path)

        # 加载tokenizer
        try:
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained(model_path / "bert")
        except Exception:
            tokenizer = None

        return {"model": model, "tokenizer": tokenizer, "type": "bert"}

    def _load_optimized(
        self, opt_type: str, model_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """加载优化后的模型"""
        if model_path is None:
            model_path = self.models_dir / "optimized" / opt_type
        else:
            model_path = Path(model_path)

        # 优化模型通常基于BERT
        model = BertClassifier.from_pretrained(model_path)

        try:
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained(model_path / "bert")
        except Exception:
            tokenizer = None

        return {"model": model, "tokenizer": tokenizer, "type": opt_type}

    def get_available_models(self) -> Dict[str, bool]:
        """
        获取可用的模型列表

        Returns:
            {模型类型: 是否可用}
        """
        available = {}

        # 检查RF
        rf_path = self.models_dir / "baseline" / "rf_tfidf" / "model.pkl"
        available["rf"] = rf_path.exists()

        # 检查FastText
        ft_path = self.models_dir / "baseline" / "fasttext" / "model.bin"
        available["fasttext"] = ft_path.exists()

        # 检查BERT
        bert_path = self.models_dir / "bert" / "best_model"
        available["bert"] = bert_path.exists()

        # 检查优化模型
        for opt_type in ["quantized", "pruned", "distilled"]:
            opt_path = self.models_dir / "optimized" / opt_type
            available[opt_type] = opt_path.exists()

        return available

    def clear_cache(self) -> None:
        """清除模型缓存"""
        self._loaded_models.clear()
