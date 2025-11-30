"""日志工具模块"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: str = "outputs/logs",
    log_name: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    设置日志配置

    Args:
        log_dir: 日志目录
        log_name: 日志文件名(不含扩展名)
        level: 日志级别
        rotation: 日志文件轮转大小
        retention: 日志保留时间
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 移除默认的处理器
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # 生成日志文件名
    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(log_dir, f"{log_name}.log")

    # 添加文件输出
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"Logger initialized. Log file: {log_path}")


def get_logger(name: str = "toumanfen"):
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logger: 日志记录器实例
    """
    return logger.bind(name=name)
