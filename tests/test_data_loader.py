"""数据加载器测试"""

import pytest
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader, parse_line, load_raw_data


class TestParseLine:
    """测试parse_line函数"""

    def test_normal_line(self):
        """测试正常行"""
        text, label = parse_line("这是一条新闻\t体育")
        assert text == "这是一条新闻"
        assert label == "体育"

    def test_empty_line(self):
        """测试空行"""
        text, label = parse_line("")
        assert text == ""
        assert label == ""

    def test_line_without_label(self):
        """测试没有标签的行"""
        text, label = parse_line("只有文本")
        assert text == ""
        assert label == ""

    def test_custom_separator(self):
        """测试自定义分隔符"""
        text, label = parse_line("文本|标签", sep="|")
        assert text == "文本"
        assert label == "标签"


class TestDataLoader:
    """测试DataLoader类"""

    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 创建测试数据
            train_data = "新闻1\t体育\n新闻2\t财经\n新闻3\t科技\n"
            test_data = "测试1\t体育\n"
            dev_data = "验证1\t财经\n"

            (tmpdir / "train.txt").write_text(train_data, encoding="utf-8")
            (tmpdir / "test.txt").write_text(test_data, encoding="utf-8")
            (tmpdir / "dev.txt").write_text(dev_data, encoding="utf-8")
            (tmpdir / "stopwords.txt").write_text("的\n了\n是\n", encoding="utf-8")
            (tmpdir / "class.txt").write_text("体育\n财经\n科技\n", encoding="utf-8")

            yield tmpdir

    def test_load_train(self, temp_data_dir):
        """测试加载训练集"""
        loader = DataLoader(temp_data_dir)
        texts, labels = loader.load_train()

        assert len(texts) == 3
        assert len(labels) == 3
        assert texts[0] == "新闻1"
        assert labels[0] == "体育"

    def test_load_stopwords(self, temp_data_dir):
        """测试加载停用词"""
        loader = DataLoader(temp_data_dir)
        stopwords = loader.load_stopwords()

        assert len(stopwords) == 3
        assert "的" in stopwords

    def test_load_class_labels(self, temp_data_dir):
        """测试加载类别标签"""
        loader = DataLoader(temp_data_dir)
        labels = loader.load_class_labels()

        assert len(labels) == 3
        assert "体育" in labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
