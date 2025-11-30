# 投满分 - 中文文本分类系统

一个完整的中文文本分类项目，支持传统机器学习模型（RF+TF-IDF、FastText）和深度学习模型（BERT），并包含模型优化技术。

## 项目特点

- **数据处理双轨制**: 基线模型和BERT模型分别预处理
- **模型多样性**: 覆盖传统ML、深度学习、模型优化
- **通用接口设计**: 评估、可视化、部署模块解耦
- **教学友好**: 包含量化、剪枝、蒸馏等优化技术展示
- **生产可用**: 完整的API部署方案

## 项目结构

```markdown
toumanfen_project/
├── configs/                # 配置文件
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理数据
│   └── vocabulary/        # 词表
├── src/                    # 源代码
│   ├── data/              # 数据处理
│   ├── features/          # 特征工程
│   ├── models/            # 模型定义
│   ├── training/          # 训练器
│   ├── evaluation/        # 评估
│   ├── visualization/     # 可视化
│   ├── deployment/        # 部署
│   └── utils/             # 工具
├── scripts/                # 运行脚本
├── tests/                  # 测试
├── models/                 # 保存的模型
├── outputs/                # 输出结果
└── docs/                   # 文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 准备数据

将原始数据放入 `data/raw/` 目录：

- `train.txt`: 训练集
- `test.txt`: 测试集
- `dev.txt`: 验证集
- `class.txt`: 类别标签
- `stopwords.txt`: 停用词表

数据格式：`文本\t标签`

### 3. 数据预处理

```bash
# 处理所有数据
python scripts/prepare_data.py --mode all

# 只处理基线模型数据
python scripts/prepare_data.py --mode baseline

# 只处理BERT数据
python scripts/prepare_data.py --mode bert
```

### 4. 模型训练

```bash
# 训练随机森林
python scripts/train_baseline.py --model rf

# 训练FastText
python scripts/train_baseline.py --model fasttext

# 训练BERT
python scripts/train_bert.py
```

### 5. 模型评估

```bash
# 评估单个模型
python scripts/evaluate.py --model bert --visualize

# 对比所有模型
python scripts/evaluate.py --compare --visualize
```

### 6. 预测

```bash
# 单条预测
python scripts/predict.py --text "央行今日进行逆回购操作" --model bert

# 批量预测
python scripts/predict.py --file input.txt --output predictions.txt
```

### 7. API服务

```bash
python src/deployment/api/app.py
```

访问 `http://localhost:5000` 查看API文档。

## 模型优化

```bash
# 量化
python scripts/optimize_model.py --method quantization --dtype int8

# 剪枝
python scripts/optimize_model.py --method pruning --sparsity 0.5

# 知识蒸馏
python scripts/optimize_model.py --method distillation
```

## API接口

### 预测

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "央行今日进行逆回购操作", "model": "bert"}'
```

### 批量预测

```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["新闻1", "新闻2"], "model": "bert"}'
```

## 配置说明

- `configs/data_config.yaml`: 数据路径和预处理配置
- `configs/model_config.yaml`: 模型超参数配置
- `configs/training_config.yaml`: 训练参数配置

## 测试

```bash
pytest tests/ -v
```

## 依赖要求

- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.20
- scikit-learn >= 1.0

## License

MIT License
