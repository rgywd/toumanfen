"""Flask/FastAPI应用模块"""

from typing import List, Optional

from flask import Flask, jsonify, request

from ..predictor import UniversalPredictor


def create_app(models_dir: str = "models", default_model: str = "bert") -> Flask:
    """
    创建Flask应用

    Args:
        models_dir: 模型目录
        default_model: 默认模型

    Returns:
        Flask应用实例
    """
    app = Flask(__name__)

    # 初始化预测器
    predictor = UniversalPredictor(models_dir=models_dir, default_model=default_model)

    @app.route("/")
    def index():
        """首页"""
        return jsonify(
            {
                "name": "投满分文本分类API",
                "version": "1.0.0",
                "endpoints": {
                    "predict": "POST /predict",
                    "batch_predict": "POST /batch_predict",
                    "models": "GET /models",
                    "health": "GET /health",
                },
            }
        )

    @app.route("/health")
    def health():
        """健康检查"""
        return jsonify({"status": "healthy"})

    @app.route("/models", methods=["GET"])
    def get_models():
        """获取可用模型列表"""
        available = predictor.get_available_models()
        return jsonify({"models": available, "default": default_model})

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        预测接口

        Request JSON:
            {
                "text": "文本内容",
                "model": "bert"  // 可选
            }

        Response JSON:
            {
                "label": "类别",
                "confidence": 0.95
            }
        """
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        model_type = data.get("model", default_model)

        try:
            result = predictor.predict_with_confidence(text, model_type)
            return jsonify(result[0])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        """
        批量预测接口

        Request JSON:
            {
                "texts": ["文本1", "文本2", ...],
                "model": "bert"  // 可选
            }

        Response JSON:
            {
                "results": [
                    {"label": "类别1", "confidence": 0.95},
                    {"label": "类别2", "confidence": 0.88},
                    ...
                ]
            }
        """
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data["texts"]
        model_type = data.get("model", default_model)

        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400

        try:
            results = predictor.predict_with_confidence(texts, model_type)
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    models_dir: str = "models",
    default_model: str = "bert",
) -> None:
    """
    启动服务器

    Args:
        host: 主机地址
        port: 端口
        debug: 调试模式
        models_dir: 模型目录
        default_model: 默认模型
    """
    app = create_app(models_dir, default_model)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
