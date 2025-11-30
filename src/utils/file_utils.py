"""文件操作工具模块"""

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在,不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    保存对象为pickle文件

    Args:
        obj: 要保存的对象
        path: 保存路径
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    加载pickle文件

    Args:
        path: 文件路径

    Returns:
        加载的对象
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    保存对象为JSON文件

    Args:
        obj: 要保存的对象
        path: 保存路径
        indent: 缩进空格数
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """
    加载JSON文件

    Args:
        path: 文件路径

    Returns:
        加载的对象
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Union[str, Path], encoding: str = "utf-8") -> List[str]:
    """
    读取文件所有行

    Args:
        path: 文件路径
        encoding: 文件编码

    Returns:
        行列表
    """
    with open(path, "r", encoding=encoding) as f:
        return [line.strip() for line in f]


def write_lines(
    lines: List[str], path: Union[str, Path], encoding: str = "utf-8"
) -> None:
    """
    写入多行到文件

    Args:
        lines: 行列表
        path: 文件路径
        encoding: 文件编码
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding=encoding) as f:
        for line in lines:
            f.write(line + "\n")


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    复制文件

    Args:
        src: 源文件路径
        dst: 目标路径

    Returns:
        目标路径
    """
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def get_file_size(path: Union[str, Path]) -> int:
    """
    获取文件大小(字节)

    Args:
        path: 文件路径

    Returns:
        文件大小
    """
    return os.path.getsize(path)


def list_files(
    directory: Union[str, Path], pattern: str = "*", recursive: bool = False
) -> List[Path]:
    """
    列出目录下的文件

    Args:
        directory: 目录路径
        pattern: 匹配模式
        recursive: 是否递归

    Returns:
        文件路径列表
    """
    directory = Path(directory)

    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))
