"""预测脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.predictor import UniversalPredictor
from src.utils.logger import setup_logger, get_logger


def main():
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File containing texts (one per line)")
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["rf", "fasttext", "bert", "quantized", "pruned", "distilled"],
        help="Model to use",
    )
    parser.add_argument("--output", type=str, help="Output file for predictions")
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Models directory"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logger(log_name="predict")
    logger = get_logger("predict")

    # 检查输入
    if not args.text and not args.file:
        parser.print_help()
        logger.error("Please provide --text or --file")
        return

    # 创建预测器
    predictor = UniversalPredictor(models_dir=args.models_dir, default_model=args.model)

    # 获取文本
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    logger.info(f"Predicting {len(texts)} texts with {args.model} model...")

    # 预测
    results = predictor.predict_with_confidence(texts, args.model)

    # 输出结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for text, result in zip(texts, results):
                f.write(f"{text}\t{result['label']}\t{result['confidence']:.4f}\n")
        logger.info(f"Predictions saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("Predictions:")
        print("=" * 60)
        for text, result in zip(texts, results):
            print(f"Text: {text[:50]}...")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print("-" * 40)
        print("=" * 60)


if __name__ == "__main__":
    main()
