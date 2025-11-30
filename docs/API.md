# API 文档

## 概述

投满分文本分类API提供RESTful接口，支持单条预测和批量预测。

## 基础信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`

## 端点

### 1. 健康检查

检查服务是否正常运行。

```python
GET /health
```

**响应示例**:

```json
{
    "status": "healthy"
}
```

### 2. 获取可用模型

获取当前可用的模型列表。

```python
GET /models
```

**响应示例**:

```json
{
    "models": {
        "rf": true,
        "fasttext": true,
        "bert": true,
        "quantized": false,
        "pruned": false,
        "distilled": false
    },
    "default": "bert"
}
```

### 3. 单条预测

对单条文本进行分类预测。

```python
POST /predict
```

**请求体**:

```json
{
    "text": "央行今日进行逆回购操作",
    "model": "bert"  // 可选，默认使用bert
}
```

**响应示例**:

```json
{
    "label": "财经",
    "confidence": 0.9523
}
```

### 4. 批量预测

对多条文本进行批量分类预测。

```python
POST /batch_predict
```

**请求体**:

```json
{
    "texts": [
        "央行今日进行逆回购操作",
        "国足在世界杯预选赛中获胜"
    ],
    "model": "bert"  // 可选
}
```

**响应示例**:

```json
{
    "results": [
        {"label": "财经", "confidence": 0.9523},
        {"label": "体育", "confidence": 0.9812}
    ]
}
```

## 错误响应

当发生错误时，API返回以下格式：

```json
{
    "error": "错误描述信息"
}
```

常见HTTP状态码：

- `400`: 请求参数错误
- `500`: 服务器内部错误

## 使用示例

### Python

```python
import requests

# 单条预测
response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "央行今日进行逆回购操作", "model": "bert"}
)
print(response.json())

# 批量预测
response = requests.post(
    "http://localhost:5000/batch_predict",
    json={"texts": ["新闻1", "新闻2"]}
)
print(response.json())
```

### cURL

```bash
# 单条预测
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "央行今日进行逆回购操作"}'

# 批量预测
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["新闻1", "新闻2"]}'
```
