"""数据准备脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessor import BaselinePreprocessor, BertPreprocessor
from src.data.vocabulary import (
    VocabularyBuilder,
    build_label_mapping,
    save_label_mapping,
)
from src.utils.config_parser import load_config
from src.utils.file_utils import save_pickle, ensure_dir
from src.utils.logger import setup_logger, get_logger
from src.utils.seed import set_seed


def prepare_baseline_data(config: dict, data_loader: DataLoader, logger):
    """准备基线模型数据"""
    logger.info("Preparing baseline data...")

    # 加载停用词
    stopwords = data_loader.load_stopwords()
    logger.info(f"Loaded {len(stopwords)} stopwords")

    # 创建预处理器
    preprocessor = BaselinePreprocessor(
        stopwords=stopwords, **config.get("preprocessing", {}).get("baseline", {})
    )

    # 加载并处理数据
    for split in ["train", "test", "dev"]:
        logger.info(f"Processing {split} set...")

        if split == "train":
            texts, labels = data_loader.load_train()
        elif split == "test":
            texts, labels = data_loader.load_test()
        else:
            texts, labels = data_loader.load_dev()

        logger.info(f"  Loaded {len(texts)} samples")

        # 预处理
        processed_texts = preprocessor.process_batch(texts)

        # 保存
        save_dir = Path(config["paths"]["processed_data_dir"]) / "baseline"
        ensure_dir(save_dir)

        data = {"texts": processed_texts, "labels": labels}
        save_pickle(data, save_dir / f"{split}_baseline.pkl")
        logger.info(f"  Saved to {save_dir / f'{split}_baseline.pkl'}")

    # 构建词表
    train_data = data_loader.load_train()
    train_texts = preprocessor.process_batch(train_data[0])

    vocab_builder = VocabularyBuilder(
        min_freq=config.get("vocabulary", {}).get("min_freq", 2),
        max_size=config.get("vocabulary", {}).get("max_size", 50000),
    )
    vocab_builder.build(train_texts)

    vocab_dir = Path(config["paths"]["vocabulary_dir"])
    vocab_builder.save(vocab_dir)
    logger.info(f"Vocabulary saved to {vocab_dir}, size: {len(vocab_builder)}")

    # 保存标签映射
    label2id, id2label = build_label_mapping(train_data[1])
    save_label_mapping(label2id, vocab_dir / "label2id.json")
    logger.info(f"Label mapping saved, {len(label2id)} classes")


def prepare_bert_data(config: dict, data_loader: DataLoader, logger):
    """准备BERT模型数据"""
    logger.info("Preparing BERT data...")

    # 创建预处理器(不使用tokenizer,只做基本清洗)
    preprocessor = BertPreprocessor(
        max_length=config.get("preprocessing", {})
        .get("bert", {})
        .get("max_length", 128)
    )

    # 加载并处理数据
    for split in ["train", "test", "dev"]:
        logger.info(f"Processing {split} set...")

        if split == "train":
            texts, labels = data_loader.load_train()
        elif split == "test":
            texts, labels = data_loader.load_test()
        else:
            texts, labels = data_loader.load_dev()

        logger.info(f"  Loaded {len(texts)} samples")

        # 基本清洗
        processed_texts = [preprocessor.clean_text(text) for text in texts]

        # 保存
        save_dir = Path(config["paths"]["processed_data_dir"]) / "bert"
        ensure_dir(save_dir)

        data = {"texts": processed_texts, "labels": labels}
        save_pickle(data, save_dir / f"{split}_bert.pkl")
        logger.info(f"  Saved to {save_dir / f'{split}_bert.pkl'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "bert", "all"],
        default="all",
        help="Data preparation mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    setup_logger(log_name="prepare_data")
    logger = get_logger("prepare_data")

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # 创建数据加载器
    data_loader = DataLoader(config["paths"]["raw_data_dir"])

    # 准备数据
    if args.mode in ["baseline", "all"]:
        prepare_baseline_data(config, data_loader, logger)

    if args.mode in ["bert", "all"]:
        prepare_bert_data(config, data_loader, logger)

    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()
