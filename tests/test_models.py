"""模型测试"""

import pytest
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline.random_forest import RFClassifier
from src.evaluation.metrics import calculate_metrics


class TestRFClassifier:
    """测试随机森林分类器"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        X_train = np.random.rand(100, 50)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.rand(20, 50)
        y_test = np.random.randint(0, 3, 20)
        return X_train, y_train, X_test, y_test

    def test_train(self, sample_data):
        """测试训练"""
        X_train, y_train, _, _ = sample_data

        model = RFClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)

        assert model.is_fitted

    def test_predict(self, sample_data):
        """测试预测"""
        X_train, y_train, X_test, _ = sample_data

        model = RFClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predict_proba(self, sample_data):
        """测试概率预测"""
        X_train, y_train, X_test, _ = sample_data

        model = RFClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape[0] == len(X_test)
        assert probas.shape[1] == 3  # 3个类别

    def test_feature_importance(self, sample_data):
        """测试特征重要性"""
        X_train, y_train, _, _ = sample_data

        model = RFClassifier(n_estimators=10, random_state=42)
        model.train(X_train, y_train)

        importance = model.get_feature_importance()
        assert len(importance) == 50  # 50个特征


class TestMetrics:
    """测试评估指标"""

    def test_calculate_metrics(self):
        """测试指标计算"""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]

        metrics = calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_perfect_predictions(self):
        """测试完美预测"""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
