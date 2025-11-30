"""路由定义模块"""

from flask import Blueprint, jsonify, request

from ..predictor import UniversalPredictor


def create_api_blueprint(
    predictor: UniversalPredictor, default_model: str = "bert"
) -> Blueprint:
    """
    创建API蓝图

    Args:
        predictor: 预测器实例
        default_model: 默认模型

    Returns:
        Flask Blueprint
    """
    api = Blueprint("api", __name__, url_prefix="/api/v1")

    @api.route("/predict", methods=["POST"])
    def predict():
        """单条预测"""
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        model_type = data.get("model", default_model)

        try:
            result = predictor.predict_with_confidence(text, model_type)
            return jsonify({"success": True, "data": result[0]})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @api.route("/batch", methods=["POST"])
    def batch_predict():
        """批量预测"""
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data["texts"]
        model_type = data.get("model", default_model)
        batch_size = data.get("batch_size", 32)

        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400

        try:
            results = predictor.predict_with_confidence(texts, model_type)
            return jsonify({"success": True, "data": results, "count": len(results)})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @api.route("/models", methods=["GET"])
    def list_models():
        """获取可用模型"""
        available = predictor.get_available_models()
        return jsonify(
            {
                "success": True,
                "data": {"available": available, "default": default_model},
            }
        )

    @api.route("/health", methods=["GET"])
    def health_check():
        """健康检查"""
        return jsonify({"success": True, "status": "healthy"})

    return api


def create_admin_blueprint() -> Blueprint:
    """
    创建管理蓝图

    Returns:
        Flask Blueprint
    """
    admin = Blueprint("admin", __name__, url_prefix="/admin")

    @admin.route("/reload", methods=["POST"])
    def reload_models():
        """重新加载模型"""
        # TODO: 实现模型重新加载
        return jsonify({"success": True, "message": "Models reloaded"})

    @admin.route("/stats", methods=["GET"])
    def get_stats():
        """获取统计信息"""
        # TODO: 实现统计信息收集
        return jsonify(
            {"success": True, "data": {"total_requests": 0, "avg_latency_ms": 0}}
        )

    return admin
