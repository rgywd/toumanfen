"""模型对比工具模块"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .metrics import calculate_metrics
from ..utils.file_utils import save_json, ensure_dir


class ModelComparator:
    """模型对比工具"""

    def __init__(self, metric_names: List[str] = None):
        """
        初始化模型对比器

        Args:
            metric_names: 要对比的指标名称
        """
        self.metric_names = metric_names or ["accuracy", "precision", "recall", "f1"]
        self.results: Dict[str, Dict] = {}

    def add_result(
        self,
        model_name: str,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        extra_info: Optional[Dict] = None,
    ) -> None:
        """
        添加模型评估结果

        Args:
            model_name: 模型名称
            y_true: 真实标签
            y_pred: 预测标签
            extra_info: 额外信息(如训练时间、模型大小等)
        """
        metrics = calculate_metrics(y_true, y_pred)

        self.results[model_name] = {"metrics": metrics, "extra_info": extra_info or {}}

    def add_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        extra_info: Optional[Dict] = None,
    ) -> None:
        """
        直接添加指标

        Args:
            model_name: 模型名称
            metrics: 指标字典
            extra_info: 额外信息
        """
        self.results[model_name] = {"metrics": metrics, "extra_info": extra_info or {}}

    def compare(self) -> Dict[str, Any]:
        """
        对比所有模型

        Returns:
            对比结果
        """
        if not self.results:
            return {}

        comparison = {
            "models": list(self.results.keys()),
            "metrics": {},
            "rankings": {},
            "best_model": {},
        }

        # 按指标整理结果
        for metric in self.metric_names:
            comparison["metrics"][metric] = {
                model_name: result["metrics"].get(metric, 0)
                for model_name, result in self.results.items()
            }

            # 排名(降序)
            sorted_models = sorted(
                comparison["metrics"][metric].items(), key=lambda x: x[1], reverse=True
            )
            comparison["rankings"][metric] = [m[0] for m in sorted_models]
            comparison["best_model"][metric] = sorted_models[0][0]

        return comparison

    def get_summary_table(self) -> str:
        """
        获取汇总表格(文本格式)

        Returns:
            表格字符串
        """
        if not self.results:
            return "No results to compare."

        # 表头
        header = ["Model"] + [m.capitalize() for m in self.metric_names]

        # 数据行
        rows = []
        for model_name, result in self.results.items():
            row = [model_name]
            for metric in self.metric_names:
                value = result["metrics"].get(metric, 0)
                row.append(f"{value:.4f}")
            rows.append(row)

        # 格式化表格
        col_widths = [
            max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))
        ]

        def format_row(row):
            return " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )

        lines = [format_row(header), "-" * (sum(col_widths) + 3 * (len(header) - 1))]
        lines.extend(format_row(row) for row in rows)

        return "\n".join(lines)

    def print_comparison(self) -> None:
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("Model Comparison Results")
        print("=" * 60)
        print(self.get_summary_table())

        comparison = self.compare()
        if comparison:
            print("\n" + "-" * 60)
            print("Best Models by Metric:")
            for metric, best in comparison["best_model"].items():
                value = comparison["metrics"][metric][best]
                print(f"  {metric.capitalize():12s}: {best} ({value:.4f})")
        print("=" * 60 + "\n")

    def save(self, save_path: Union[str, Path]) -> None:
        """
        保存对比结果

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        ensure_dir(save_path.parent)

        data = {"results": self.results, "comparison": self.compare()}
        save_json(data, save_path)

    def get_best_model(self, metric: str = "f1") -> str:
        """
        获取最佳模型名称

        Args:
            metric: 评估指标

        Returns:
            最佳模型名称
        """
        comparison = self.compare()
        return comparison.get("best_model", {}).get(metric, "")
