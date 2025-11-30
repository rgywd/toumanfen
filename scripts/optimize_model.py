"""模型优化脚本(教学用)"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models.bert.bert_classifier import BertClassifier
from src.models.optimization.quantization import ModelQuantizer, compare_model_size
from src.models.optimization.pruning import ModelPruner, get_model_sparsity
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.file_utils import ensure_dir


def quantize_model(args, logger):
    """量化模型"""
    logger.info("Loading BERT model for quantization...")

    model = BertClassifier.from_pretrained("models/bert/best_model")

    # 创建量化器
    quantizer = ModelQuantizer(dtype=args.dtype, dynamic=True)

    logger.info(f"Quantizing model to {args.dtype}...")
    quantized_model = quantizer.quantize(model)

    # 比较大小
    size_info = compare_model_size(model, quantized_model)
    logger.info(f"Original size: {size_info['original_size_mb']:.2f} MB")
    logger.info(f"Quantized size: {size_info['quantized_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {size_info['compression_ratio']:.2f}x")

    # 保存
    save_dir = Path("models/optimized/quantized")
    ensure_dir(save_dir)
    quantizer.save(quantized_model, save_dir / "model.pt")
    logger.info(f"Quantized model saved to {save_dir}")


def prune_model(args, logger):
    """剪枝模型"""
    logger.info("Loading BERT model for pruning...")

    model = BertClassifier.from_pretrained("models/bert/best_model")

    # 原始稀疏度
    original_sparsity = get_model_sparsity(model)
    logger.info(f"Original sparsity: {original_sparsity['sparsity_percent']}")

    # 创建剪枝器
    pruner = ModelPruner(
        sparsity=args.sparsity, method=args.method, structured=args.structured
    )

    logger.info(f"Pruning model with sparsity={args.sparsity}...")
    pruned_model = pruner.prune(model)

    # 剪枝后稀疏度
    new_sparsity = get_model_sparsity(pruned_model)
    logger.info(f"New sparsity: {new_sparsity['sparsity_percent']}")

    # 移除剪枝的重参数化
    pruner.remove_pruning(pruned_model)

    # 保存
    save_dir = Path("models/optimized/pruned")
    ensure_dir(save_dir)
    pruner.save(pruned_model, save_dir / "model.pt")
    logger.info(f"Pruned model saved to {save_dir}")


def distill_model(args, logger):
    """知识蒸馏"""
    logger.info("Knowledge distillation...")
    logger.info(
        "Note: This is a demonstration. Full implementation requires training data."
    )

    # 加载教师模型
    teacher = BertClassifier.from_pretrained("models/bert/best_model")

    # 创建学生模型(可以是更小的模型)
    from src.models.bert.config import BertConfig

    student_config = BertConfig(
        model_name="bert-base-chinese",  # 或更小的模型
        num_labels=teacher.config.num_labels,
    )
    student = BertClassifier(config=student_config)

    logger.info(f"Teacher model: {teacher.config.model_name}")
    logger.info(f"Student model: {student_config.model_name}")
    logger.info(
        "To perform distillation, use BertTrainer with DistillationTrainer wrapper"
    )

    # 保存学生模型(初始化状态)
    save_dir = Path("models/optimized/distilled")
    ensure_dir(save_dir)
    logger.info(f"Student model directory: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Optimize models")
    parser.add_argument(
        "--method",
        type=str,
        choices=["quantization", "pruning", "distillation"],
        required=True,
        help="Optimization method",
    )

    # 量化参数
    parser.add_argument(
        "--dtype",
        type=str,
        default="int8",
        choices=["int8", "float16"],
        help="Quantization data type",
    )

    # 剪枝参数
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Pruning sparsity (0-1)"
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="magnitude",
        choices=["magnitude", "random"],
        help="Pruning method",
    )
    parser.add_argument(
        "--structured", action="store_true", help="Use structured pruning"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logger(log_name=f"optimize_{args.method}")
    logger = get_logger(f"optimize_{args.method}")

    logger.info(f"Optimization method: {args.method}")

    if args.method == "quantization":
        quantize_model(args, logger)
    elif args.method == "pruning":
        prune_model(args, logger)
    elif args.method == "distillation":
        distill_model(args, logger)

    logger.info("Optimization completed!")


if __name__ == "__main__":
    main()
