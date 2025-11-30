"""模型剪枝模块(教学用)"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelPruner:
    """
    模型剪枝器
    支持结构化和非结构化剪枝
    """

    def __init__(
        self, sparsity: float = 0.5, method: str = "magnitude", structured: bool = False
    ):
        """
        初始化剪枝器

        Args:
            sparsity: 稀疏度(剪枝比例)
            method: 剪枝方法 ("magnitude", "random", "l1")
            structured: 是否结构化剪枝
        """
        self.sparsity = sparsity
        self.method = method
        self.structured = structured

    def prune(
        self, model: nn.Module, save_path: Optional[Union[str, Path]] = None
    ) -> nn.Module:
        """
        剪枝模型

        Args:
            model: 原始模型
            save_path: 保存路径

        Returns:
            剪枝后的模型
        """
        # 获取需要剪枝的层
        layers_to_prune = self._get_layers_to_prune(model)

        # 应用剪枝
        for layer, param_name in layers_to_prune:
            if self.structured:
                self._structured_prune(layer, param_name)
            else:
                self._unstructured_prune(layer, param_name)

        # 保存
        if save_path:
            self.save(model, save_path)

        return model

    def _get_layers_to_prune(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """获取需要剪枝的层"""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((module, "weight"))
            elif isinstance(module, nn.Conv2d):
                layers.append((module, "weight"))
        return layers

    def _unstructured_prune(self, layer: nn.Module, param_name: str) -> None:
        """非结构化剪枝(元素级)"""
        if self.method == "magnitude":
            prune.l1_unstructured(layer, param_name, amount=self.sparsity)
        elif self.method == "random":
            prune.random_unstructured(layer, param_name, amount=self.sparsity)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _structured_prune(self, layer: nn.Module, param_name: str) -> None:
        """结构化剪枝(通道/过滤器级)"""
        if self.method in ["magnitude", "l1"]:
            prune.ln_structured(
                layer,
                param_name,
                amount=self.sparsity,
                n=1,  # L1范数
                dim=0,  # 按输出通道剪枝
            )
        elif self.method == "random":
            prune.random_structured(layer, param_name, amount=self.sparsity, dim=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """
        移除剪枝的重参数化,使剪枝永久化

        Args:
            model: 剪枝后的模型

        Returns:
            永久化剪枝的模型
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass  # 该层未被剪枝
        return model

    def save(self, model: nn.Module, save_path: Union[str, Path]) -> None:
        """保存剪枝模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    def load(self, model: nn.Module, load_path: Union[str, Path]) -> nn.Module:
        """加载剪枝模型"""
        model.load_state_dict(torch.load(load_path))
        return model


def get_model_sparsity(model: nn.Module) -> dict:
    """
    计算模型稀疏度

    Args:
        model: 模型

    Returns:
        稀疏度信息
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0

    return {
        "total_params": total_params,
        "zero_params": zero_params,
        "sparsity": sparsity,
        "sparsity_percent": f"{sparsity * 100:.2f}%",
    }
