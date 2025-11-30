"""通用评估器模块"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .metrics import (
    calculate_metrics,
    get_classification_report,
    get_confusion_matrix,
    calculate_per_class_metrics,
)
from ..utils.file_utils import save_json, ensure_dir


class Evaluator:
    """通用评估器"""

    def __init__(
        self, metrics: List[str] = None, average: str = "macro", per_class: bool = True
    ):
        """
        初始化评估器

        Args:
            metrics: 要计算的指标列表
            average: 平均方式
            per_class: 是否计算每个类别的指标
        """
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1"]
        self.average = average
        self.per_class = per_class

    def evaluate(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        labels: Optional[List] = None,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        评估预测结果

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            label_names: 标签名称

        Returns:
            评估结果字典
        """
        result = {}

        # 基础指标
        basic_metrics = calculate_metrics(
            y_true, y_pred, average=self.average, labels=labels
        )
        result["metrics"] = basic_metrics

        # 分类报告
        result["classification_report"] = get_classification_report(
            y_true, y_pred, labels=labels, target_names=label_names, output_dict=True
        )

        # 混淆矩阵
        result["confusion_matrix"] = get_confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()

        # 每个类别的指标
        if self.per_class:
            result["per_class_metrics"] = calculate_per_class_metrics(
                y_true, y_pred, labels=labels
            )

        return result

    def evaluate_model(
        self,
        model: Any,
        X: Any,
        y_true: Union[List, np.ndarray],
        predict_fn: Optional[Callable] = None,
        labels: Optional[List] = None,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        评估模型

        Args:
            model: 模型实例
            X: 输入数据
            y_true: 真实标签
            predict_fn: 自定义预测函数
            labels: 标签列表
            label_names: 标签名称

        Returns:
            评估结果字典
        """
        # 获取预测结果
        if predict_fn is not None:
            y_pred = predict_fn(model, X)
        elif hasattr(model, "predict"):
            y_pred = model.predict(X)
        else:
            raise ValueError("Model must have predict method or provide predict_fn")

        return self.evaluate(y_true, y_pred, labels, label_names)

    def save_results(
        self, results: Dict[str, Any], save_path: Union[str, Path]
    ) -> None:
        """
        保存评估结果

        Args:
            results: 评估结果
            save_path: 保存路径
        """
        save_path = Path(save_path)
        ensure_dir(save_path.parent)

        # 转换numpy数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        results = convert_numpy(results)
        save_json(results, save_path)


def quick_evaluate(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    print_report: bool = True,
) -> Dict[str, float]:
    """
    快速评估

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        print_report: 是否打印报告

    Returns:
        指标字典
    """
    metrics = calculate_metrics(y_true, y_pred)

    if print_report:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for name, value in metrics.items():
            print(f"{name.capitalize():12s}: {value:.4f}")
        print("=" * 50 + "\n")

    return metrics
