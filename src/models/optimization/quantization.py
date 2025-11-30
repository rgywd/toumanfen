"""模型量化模块(教学用)"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class ModelQuantizer:
    """
    模型量化器
    支持动态量化和静态量化
    """

    def __init__(self, dtype: str = "int8", dynamic: bool = True):
        """
        初始化量化器

        Args:
            dtype: 量化数据类型 ("int8" 或 "float16")
            dynamic: 是否使用动态量化
        """
        self.dtype = dtype
        self.dynamic = dynamic

    def quantize(
        self, model: nn.Module, save_path: Optional[Union[str, Path]] = None
    ) -> nn.Module:
        """
        量化模型

        Args:
            model: 原始模型
            save_path: 保存路径

        Returns:
            量化后的模型
        """
        if self.dtype == "int8":
            quantized_model = self._quantize_int8(model)
        elif self.dtype == "float16":
            quantized_model = self._quantize_fp16(model)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        if save_path:
            self.save(quantized_model, save_path)

        return quantized_model

    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """INT8动态量化"""
        model.eval()

        if self.dynamic:
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8  # 量化Linear层
            )
        else:
            # 静态量化(需要校准)
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model, inplace=True)
            # 注意: 静态量化需要用代表性数据校准
            quantized_model = torch.quantization.convert(model, inplace=False)

        return quantized_model

    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """FP16量化(半精度)"""
        return model.half()

    def save(self, model: nn.Module, save_path: Union[str, Path]) -> None:
        """
        保存量化模型

        Args:
            model: 量化模型
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    def load(self, model: nn.Module, load_path: Union[str, Path]) -> nn.Module:
        """
        加载量化模型

        Args:
            model: 模型架构
            load_path: 加载路径

        Returns:
            加载权重后的模型
        """
        model.load_state_dict(torch.load(load_path))
        return model


def compare_model_size(original_model: nn.Module, quantized_model: nn.Module) -> dict:
    """
    比较模型大小

    Args:
        original_model: 原始模型
        quantized_model: 量化模型

    Returns:
        大小对比信息
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(original_model.state_dict(), f.name)
        original_size = os.path.getsize(f.name)
        os.unlink(f.name)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(quantized_model.state_dict(), f.name)
        quantized_size = os.path.getsize(f.name)
        os.unlink(f.name)

    return {
        "original_size_mb": original_size / (1024 * 1024),
        "quantized_size_mb": quantized_size / (1024 * 1024),
        "compression_ratio": original_size / quantized_size,
    }
