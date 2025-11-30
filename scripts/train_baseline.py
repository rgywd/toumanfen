"""基线模型训练脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.baseline_trainer import BaselineTrainer
from src.utils.config_parser import load_config
from src.utils.file_utils import load_pickle
from src.utils.logger import setup_logger, get_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "fasttext"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cv", type=int, default=0, help="Cross-validation folds (0 to disable)"
    )

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    setup_logger(log_name=f"train_{args.model}")
    logger = get_logger(f"train_{args.model}")

    # 加载配置
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    logger.info(f"Training {args.model} model...")

    # 加载数据
    data_dir = Path(data_config["paths"]["processed_data_dir"]) / "baseline"

    train_data = load_pickle(data_dir / "train_baseline.pkl")
    dev_data = load_pickle(data_dir / "dev_baseline.pkl")

    train_texts = train_data["texts"]
    train_labels = train_data["labels"]
    dev_texts = dev_data["texts"]
    dev_labels = dev_data["labels"]

    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Dev samples: {len(dev_texts)}")

    # 获取模型配置
    if args.model == "rf":
        model_kwargs = model_config["baseline"]["random_forest"]
    else:
        model_kwargs = model_config["baseline"]["fasttext"]

    # 创建训练器
    trainer = BaselineTrainer(
        model_type=args.model, save_dir="models/baseline", **model_kwargs
    )

    # 训练
    result = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=dev_texts,
        val_labels=dev_labels,
        cross_validation=args.cv,
    )

    # 打印结果
    logger.info("=" * 50)
    logger.info("Training Results:")
    logger.info("=" * 50)

    if "cv_mean" in result:
        logger.info(
            f"CV F1 Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})"
        )

    logger.info("Train Metrics:")
    for k, v in result["train_metrics"].items():
        logger.info(f"  {k}: {v:.4f}")

    if "val_metrics" in result:
        logger.info("Validation Metrics:")
        for k, v in result["val_metrics"].items():
            logger.info(f"  {k}: {v:.4f}")

    logger.info("=" * 50)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
