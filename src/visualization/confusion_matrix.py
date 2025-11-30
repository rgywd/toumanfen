"""混淆矩阵可视化模块"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
    annotate: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        cmap: 颜色映射
        normalize: 是否归一化
        annotate: 是否显示数值
        fmt: 数值格式
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # 处理除零情况

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=figsize)

    # 使用seaborn绘制热力图
    sns.heatmap(
        cm,
        annot=annotate,
        fmt=fmt if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        square=True,
        linewidths=0.5,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 旋转标签
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix_from_predictions(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    **kwargs
) -> None:
    """
    从预测结果绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
        normalize: 是否归一化
        **kwargs: 其他参数
    """
    from sklearn.metrics import confusion_matrix

    # 获取类别
    if class_names is None:
        classes = sorted(set(y_true) | set(y_pred))
        class_names = [str(c) for c in classes]
    else:
        classes = class_names

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=title,
        save_path=save_path,
        normalize=normalize,
        **kwargs
    )


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Normalized Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    绘制归一化混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        title: 图表标题
        save_path: 保存路径
        **kwargs: 其他参数
    """
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=title,
        save_path=save_path,
        normalize=True,
        **kwargs
    )
