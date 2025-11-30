"""评估脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import Evaluator, quick_evaluate
from src.evaluation.model_comparator import ModelComparator
from src.deployment.predictor import UniversalPredictor
from src.utils.config_parser import load_config
from src.utils.file_utils import load_pickle, save_json
from src.utils.logger import setup_logger, get_logger
from src.visualization.plotter import plot_metrics, plot_comparison
from src.visualization.confusion_matrix import plot_confusion_matrix_from_predictions


def evaluate_single_model(args, logger):
    """评估单个模型"""
    # 加载配置
    data_config = load_config(args.data_config)

    # 加载测试数据
    if args.model in ["rf", "fasttext"]:
        data_dir = Path(data_config["paths"]["processed_data_dir"]) / "baseline"
    else:
        data_dir = Path(data_config["paths"]["processed_data_dir"]) / "bert"

    test_data = load_pickle(
        data_dir
        / f"test_{'baseline' if args.model in ['rf', 'fasttext'] else 'bert'}.pkl"
    )
    test_texts = test_data["texts"]
    test_labels = test_data["labels"]

    logger.info(f"Loaded {len(test_texts)} test samples")

    # 创建预测器
    predictor = UniversalPredictor(models_dir="models", default_model=args.model)

    # 预测
    logger.info(f"Predicting with {args.model} model...")
    predictions = predictor.predict(test_texts, args.model)

    # 评估
    evaluator = Evaluator(per_class=True)
    results = evaluator.evaluate(test_labels, predictions)

    # 打印结果
    logger.info("=" * 50)
    logger.info(f"Evaluation Results for {args.model.upper()}")
    logger.info("=" * 50)

    for metric, value in results["metrics"].items():
        logger.info(f"{metric.capitalize():12s}: {value:.4f}")

    # 保存结果
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.save_results(results, output_dir / f"{args.model}_results.json")
    logger.info(f"Results saved to {output_dir / f'{args.model}_results.json'}")

    # 可视化
    if args.visualize:
        viz_dir = Path("outputs/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 指标图
        plot_metrics(
            results["metrics"],
            title=f"{args.model.upper()} Model Metrics",
            save_path=viz_dir / f"{args.model}_metrics.png",
        )

        # 混淆矩阵
        plot_confusion_matrix_from_predictions(
            test_labels,
            predictions,
            title=f"{args.model.upper()} Confusion Matrix",
            save_path=viz_dir / f"{args.model}_confusion.png",
        )

    return results


def compare_models(args, logger):
    """对比所有模型"""
    # 加载配置
    data_config = load_config(args.data_config)

    # 创建比较器
    comparator = ModelComparator()

    # 加载测试数据(基线)
    baseline_dir = Path(data_config["paths"]["processed_data_dir"]) / "baseline"
    baseline_test = load_pickle(baseline_dir / "test_baseline.pkl")

    # 加载测试数据(BERT)
    bert_dir = Path(data_config["paths"]["processed_data_dir"]) / "bert"
    bert_test = load_pickle(bert_dir / "test_bert.pkl")

    # 创建预测器
    predictor = UniversalPredictor(models_dir="models")
    available = predictor.get_available_models()

    model_metrics = {}

    for model_type, is_available in available.items():
        if not is_available:
            logger.warning(f"Model {model_type} not available, skipping...")
            continue

        logger.info(f"Evaluating {model_type}...")

        # 选择对应的测试数据
        if model_type in ["rf", "fasttext"]:
            test_texts = baseline_test["texts"]
            test_labels = baseline_test["labels"]
        else:
            test_texts = bert_test["texts"]
            test_labels = bert_test["labels"]

        try:
            predictions = predictor.predict(test_texts, model_type)
            comparator.add_result(model_type, test_labels, predictions)

            # 记录指标用于可视化
            from src.evaluation.metrics import calculate_metrics

            model_metrics[model_type] = calculate_metrics(test_labels, predictions)

        except Exception as e:
            logger.error(f"Failed to evaluate {model_type}: {e}")

    # 打印对比结果
    comparator.print_comparison()

    # 保存结果
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparator.save(output_dir / "model_comparison.json")
    logger.info(f"Comparison saved to {output_dir / 'model_comparison.json'}")

    # 可视化对比
    if args.visualize and model_metrics:
        viz_dir = Path("outputs/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison(
            model_metrics,
            title="Model Comparison",
            save_path=viz_dir / "model_comparison.png",
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "fasttext", "bert"],
        help="Model to evaluate (single model mode)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare all available models"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logger(log_name="evaluate")
    logger = get_logger("evaluate")

    if args.compare:
        compare_models(args, logger)
    elif args.model:
        evaluate_single_model(args, logger)
    else:
        parser.print_help()
        logger.error("Please specify --model or --compare")


if __name__ == "__main__":
    main()
