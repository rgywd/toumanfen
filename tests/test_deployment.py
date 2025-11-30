"""部署模块测试"""

import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.model_loader import UniversalModelLoader


class TestUniversalModelLoader:
    """测试通用模型加载器"""

    @pytest.fixture
    def loader(self):
        """创建加载器"""
        return UniversalModelLoader(models_dir="models")

    def test_get_available_models(self, loader):
        """测试获取可用模型"""
        available = loader.get_available_models()

        assert isinstance(available, dict)
        assert "rf" in available
        assert "fasttext" in available
        assert "bert" in available

    def test_unsupported_model_type(self, loader):
        """测试不支持的模型类型"""
        with pytest.raises(ValueError):
            loader.load_model("unknown_model")

    def test_clear_cache(self, loader):
        """测试清除缓存"""
        loader.clear_cache()
        assert len(loader._loaded_models) == 0


class TestAPI:
    """测试API"""

    def test_create_app(self):
        """测试创建Flask应用"""
        from src.deployment.api.app import create_app

        app = create_app(models_dir="models")
        assert app is not None

    def test_health_endpoint(self):
        """测试健康检查端点"""
        from src.deployment.api.app import create_app

        app = create_app(models_dir="models")
        client = app.test_client()

        response = client.get("/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "healthy"

    def test_models_endpoint(self):
        """测试模型列表端点"""
        from src.deployment.api.app import create_app

        app = create_app(models_dir="models")
        client = app.test_client()

        response = client.get("/models")
        assert response.status_code == 200

        data = response.get_json()
        assert "models" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
