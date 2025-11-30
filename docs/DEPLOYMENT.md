# 部署文档

## 部署方式

### 1. 本地部署

#### 启动Flask服务

```bash
python src/deployment/api/app.py
```

默认运行在 `http://localhost:5000`

#### 自定义配置

```python
from src.deployment.api.app import run_server

run_server(
    host="0.0.0.0",
    port=8080,
    debug=False,
    models_dir="models",
    default_model="bert"
)
```

### 2. Docker部署

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 5000

CMD ["python", "src/deployment/api/app.py"]
```

#### 构建和运行

```bash
docker build -t toumanfen:latest .
docker run -p 5000:5000 toumanfen:latest
```

### 3. Gunicorn部署

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 "src.deployment.api.app:create_app()"
```

### 4. Nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 模型管理

### 模型加载

```python
from src.deployment.model_loader import UniversalModelLoader

loader = UniversalModelLoader(models_dir="models")

# 加载特定模型
model_data = loader.load_model("bert")

# 查看可用模型
available = loader.get_available_models()
```

### 预测器使用

```python
from src.deployment.predictor import UniversalPredictor

predictor = UniversalPredictor(
    models_dir="models",
    default_model="bert"
)

# 单条预测
result = predictor.predict("测试文本", model_type="bert")

# 带置信度预测
results = predictor.predict_with_confidence(["文本1", "文本2"])

# 批量预测
predictions = predictor.batch_predict(texts, batch_size=32)
```

## 性能优化

### 1. 模型缓存

模型加载后会自动缓存，避免重复加载。

### 2. 批量处理

对于大量请求，使用批量预测接口提高效率。

### 3. GPU加速

BERT模型支持GPU加速：

```python
predictor = UniversalPredictor(
    models_dir="models",
    device="cuda"  # 使用GPU
)
```

### 4. 使用优化模型

部署量化或剪枝后的模型以提高推理速度：

```python
predictor.predict(text, model_type="quantized")
```

## 监控和日志

### 日志配置

```python
from src.utils.logger import setup_logger

setup_logger(
    log_dir="outputs/logs",
    log_name="api",
    level="INFO"
)
```

### 健康检查

```bash
curl http://localhost:5000/health
```

## 安全建议

1. **生产环境关闭debug模式**
2. **使用HTTPS**
3. **添加API认证**
4. **限制请求频率**
5. **输入验证和清洗**

## 常见问题

### Q: 模型加载慢？

A: 首次加载需要时间，之后会缓存。可以在启动时预加载常用模型。

### Q: 内存不足？

A: 使用量化模型或减少同时加载的模型数量。

### Q: GPU内存不足？

A: 减小batch_size或使用CPU进行推理。
