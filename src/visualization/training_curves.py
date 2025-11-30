"""训练曲线可视化模块"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_training_curves(
    train_history: List[Dict[str, float]],
    val_history: Optional[List[Dict[str, float]]] = None,
    metrics: List[str] = None,
    title: str = "Training Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    绘制训练曲线

    Args:
        train_history: 训练历史
        val_history: 验证历史
        metrics: 要绘制的指标
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    if metrics is None:
        metrics = ["loss"]
        if train_history and "f1" in train_history[0]:
            metrics.append("f1")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # 训练数据
        train_values = [h.get(metric, 0) for h in train_history]
        epochs = range(1, len(train_values) + 1)
        ax.plot(epochs, train_values, "b-", label=f"Train {metric}", linewidth=2)

        # 验证数据
        if val_history:
            val_values = [h.get(metric, 0) for h in val_history]
            ax.plot(epochs, val_values, "r-", label=f"Val {metric}", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f"{metric.capitalize()} Curve", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Loss Curve",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    绘制损失曲线

    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2, marker="o")

    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2, marker="s")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_learning_rate_curve(
    learning_rates: List[float],
    title: str = "Learning Rate Schedule",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    绘制学习率曲线

    Args:
        learning_rates: 学习率列表
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = range(len(learning_rates))
    ax.plot(steps, learning_rates, "g-", linewidth=2)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # 科学计数法
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_metrics_curves(
    history: Dict[str, List[float]],
    title: str = "Metrics Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    绘制多个指标曲线

    Args:
        history: {指标名: 值列表}
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(history)))

    for (metric_name, values), color in zip(history.items(), colors):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=metric_name, linewidth=2, color=color)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
