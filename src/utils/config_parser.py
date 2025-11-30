"""配置解析工具模块"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigParser:
    """配置解析器"""

    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置解析器

        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_name: 配置文件名(不含扩展名)

        Returns:
            配置字典
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            config_name: 配置文件名
            key: 配置键(支持点号分隔的嵌套键)
            default: 默认值

        Returns:
            配置值
        """
        config = self.load(config_name)

        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def load_all(self) -> Dict[str, Dict]:
        """加载所有配置文件"""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            self.load(config_name)
        return self._configs


def load_config(
    config_path: Union[str, Path], encoding: str = "utf-8"
) -> Dict[str, Any]:
    """
    加载单个配置文件

    Args:
        config_path: 配置文件路径
        encoding: 文件编码

    Returns:
        配置字典
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding=encoding) as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    合并配置字典(深度合并)

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
