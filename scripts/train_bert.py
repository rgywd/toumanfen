"""BERT模型训练脚本"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.data.vocabulary import load_label_mapping
from src.models.bert.bert_classifier import BertClassifier
from src.models.bert.config import BertConfig
from src.training.bert_trainer import BertTrainer
from src.utils.config_parser import load_config
from src.utils.file_utils import load_pickle
from src.utils.logger import setup_logger, get_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train BERT model")
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
        "--epochs", type=int, default=None, help="Override training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    setup_logger(log_name="train_bert")
    logger = get_logger("train_bert")

    # 加载配置
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    bert_model_config = model_config["bert"]
    bert_training_config = training_config["bert"]

    # 覆盖参数
    if args.epochs:
        bert_training_config["epochs"] = args.epochs
    if args.batch_size:
        bert_training_config["batch_size"] = args.batch_size
    if args.lr:
        bert_training_config["learning_rate"] = args.lr

    logger.info("Training BERT model...")
    logger.info(f"Model: {bert_model_config['model_name']}")
    logger.info(f"Epochs: {bert_training_config['epochs']}")
    logger.info(f"Batch size: {bert_training_config['batch_size']}")
    logger.info(f"Learning rate: {bert_training_config['learning_rate']}")

    # 加载数据
    data_dir = Path(data_config["paths"]["processed_data_dir"]) / "bert"
    vocab_dir = Path(data_config["paths"]["vocabulary_dir"])

    train_data = load_pickle(data_dir / "train_bert.pkl")
    dev_data = load_pickle(data_dir / "dev_bert.pkl")

    train_texts = train_data["texts"]
    train_labels = train_data["labels"]
    dev_texts = dev_data["texts"]
    dev_labels = dev_data["labels"]

    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Dev samples: {len(dev_texts)}")

    # 加载标签映射
    label2id = load_label_mapping(vocab_dir / "label2id.json")
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    logger.info(f"Number of classes: {num_labels}")

    # 加载tokenizer
    try:
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(bert_model_config["model_name"])
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # 编码数据
    logger.info("Encoding data...")

    train_encodings = tokenizer(
        train_texts,
        max_length=bert_model_config["max_length"],
        truncation=True,
        padding="max_length",
    )
    dev_encodings = tokenizer(
        dev_texts,
        max_length=bert_model_config["max_length"],
        truncation=True,
        padding="max_length",
    )

    # 转换标签
    train_label_ids = [label2id[label] for label in train_labels]
    dev_label_ids = [label2id[label] for label in dev_labels]

    # 创建Dataset
    train_dataset = TextDataset(train_encodings, train_label_ids)
    dev_dataset = TextDataset(dev_encodings, dev_label_ids)

    # 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bert_training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["common"]["num_workers"],
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=bert_training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["common"]["num_workers"],
    )

    # 创建模型
    config = BertConfig(
        model_name=bert_model_config["model_name"],
        num_labels=num_labels,
        max_length=bert_model_config["max_length"],
        hidden_dropout_prob=bert_model_config["hidden_dropout_prob"],
        classifier_dropout=bert_model_config.get("classifier_dropout", 0.1),
    )
    model = BertClassifier(config=config)

    # 创建训练器
    trainer = BertTrainer(
        model=model,
        save_dir="models/bert",
        device=training_config["common"]["device"],
        epochs=bert_training_config["epochs"],
        learning_rate=bert_training_config["learning_rate"],
        weight_decay=bert_training_config["weight_decay"],
        warmup_ratio=bert_training_config["warmup_ratio"],
        max_grad_norm=bert_training_config["max_grad_norm"],
        fp16=bert_training_config["fp16"],
        early_stopping_patience=bert_training_config["early_stopping"]["patience"],
        logging_steps=bert_training_config["logging"]["logging_steps"],
    )

    # 训练
    result = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=dev_dataloader,
        id2label=id2label,
    )

    # 打印结果
    logger.info("=" * 50)
    logger.info("Training Results:")
    logger.info("=" * 50)

    if result["best_metrics"]:
        logger.info("Best Validation Metrics:")
        for k, v in result["best_metrics"].items():
            logger.info(f"  {k}: {v:.4f}")

    logger.info("=" * 50)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
