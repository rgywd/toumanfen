"""图表绘制模块"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metrics(
    metrics: Dict[str, float],
    title: str = "Model Metrics",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
    color: str = "#4CAF50",
) -> None:
    """
    绘制性能指标柱状图

    Args:
        metrics: 指标字典
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
        color: 柱子颜色
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color=color, edgecolor="white", linewidth=1.2)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    绘制模型对比图

    Args:
        model_metrics: {模型名: {指标名: 值}}
        metric_names: 要展示的指标列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    if metric_names is None:
        metric_names = ["accuracy", "precision", "recall", "f1"]

    model_names = list(model_metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_bar_chart(
    data: Dict[str, float],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
    horizontal: bool = False,
) -> None:
    """
    通用柱状图

    Args:
        data: 数据字典
        xlabel: X轴标签
        ylabel: Y轴标签
        title: 标题
        save_path: 保存路径
        figsize: 图表大小
        horizontal: 是否水平
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(data.keys())
    values = list(data.values())

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    if horizontal:
        ax.barh(names, values, color=colors)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(names, values, color=colors)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
