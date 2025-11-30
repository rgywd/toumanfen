"""预处理器测试"""

import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import BaselinePreprocessor, BertPreprocessor


class TestBaselinePreprocessor:
    """测试BaselinePreprocessor"""

    @pytest.fixture
    def preprocessor(self):
        """创建预处理器"""
        stopwords = {"的", "了", "是", "在"}
        return BaselinePreprocessor(
            stopwords=stopwords, remove_stopwords=True, remove_punctuation=True
        )

    def test_clean_text(self, preprocessor):
        """测试文本清洗"""
        text = "这是一段测试文本！！！"
        cleaned = preprocessor.clean_text(text)
        assert "！" not in cleaned

    def test_tokenize(self, preprocessor):
        """测试分词"""
        text = "我爱北京天安门"
        words = preprocessor.tokenize(text)
        assert len(words) > 1

    def test_filter_stopwords(self, preprocessor):
        """测试停用词过滤"""
        words = ["我", "是", "一个", "学生"]
        filtered = preprocessor.filter_words(words)
        assert "是" not in filtered

    def test_process(self, preprocessor):
        """测试完整处理流程"""
        text = "这是一段测试文本"
        processed = preprocessor.process(text)
        assert isinstance(processed, str)
        assert "是" not in processed.split()

    def test_process_batch(self, preprocessor):
        """测试批量处理"""
        texts = ["文本1", "文本2", "文本3"]
        processed = preprocessor.process_batch(texts)
        assert len(processed) == 3


class TestBertPreprocessor:
    """测试BertPreprocessor"""

    @pytest.fixture
    def preprocessor(self):
        """创建预处理器"""
        return BertPreprocessor(max_length=128)

    def test_clean_text(self, preprocessor):
        """测试文本清洗"""
        text = "  多余   空格   测试  "
        cleaned = preprocessor.clean_text(text)
        assert "  " not in cleaned

    def test_process_without_tokenizer(self, preprocessor):
        """测试无tokenizer时的处理"""
        with pytest.raises(ValueError):
            preprocessor.process("测试文本")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
