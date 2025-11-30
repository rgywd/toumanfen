"""评估指标计算模块"""

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    average: str = "macro",
    labels: Optional[List] = None,
) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ("macro", "micro", "weighted")
        labels: 标签列表

    Returns:
        指标字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, labels=labels, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, labels=labels, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0),
    }
    return metrics


def get_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    output_dict: bool = True,
) -> Union[str, Dict]:
    """
    获取分类报告

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表
        target_names: 标签名称
        output_dict: 是否输出字典格式

    Returns:
        分类报告
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def get_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    获取混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表
        normalize: 归一化方式 ("true", "pred", "all", None)

    Returns:
        混淆矩阵
    """
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def calculate_per_class_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List] = None,
) -> Dict[str, Dict[str, float]]:
    """
    计算每个类别的指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签列表

    Returns:
        每个类别的指标字典
    """
    report = get_classification_report(y_true, y_pred, labels=labels, output_dict=True)

    # 移除汇总指标
    per_class = {
        k: v
        for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    return per_class


def top_k_accuracy(
    y_true: Union[List, np.ndarray], y_proba: np.ndarray, k: int = 5
) -> float:
    """
    计算Top-K准确率

    Args:
        y_true: 真实标签(整数索引)
        y_proba: 预测概率矩阵
        k: K值

    Returns:
        Top-K准确率
    """
    y_true = np.array(y_true)
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]

    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1

    return correct / len(y_true)
