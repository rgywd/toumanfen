# 模型文档

## 模型概述

本项目支持以下文本分类模型：

| 模型        | 类型         | 特点                 |
| ----------- | ------------ | -------------------- |
| RF + TF-IDF | 传统ML       | 快速训练，可解释性强 |
| FastText    | 浅层神经网络 | 训练快，支持n-gram   |
| BERT        | 深度学习     | 效果最好，需要GPU    |

## 1. 随机森林 + TF-IDF

### 原理

- **TF-IDF**: 将文本转换为词频-逆文档频率向量
- **随机森林**: 集成多个决策树进行分类

### 配置参数

```yaml
random_forest:
  n_estimators: 100      # 树的数量
  max_depth: 50          # 最大深度
  min_samples_split: 2   # 分裂最小样本数
  
tfidf:
  max_features: 10000    # 最大特征数
  ngram_range: [1, 2]    # n-gram范围
```

### 使用示例

```python
from src.models.baseline.random_forest import RFClassifier
from src.features.tfidf_extractor import TfidfExtractor

# 特征提取
tfidf = TfidfExtractor(max_features=10000)
X_train = tfidf.fit_transform(train_texts)

# 训练
model = RFClassifier(n_estimators=100)
model.train(X_train, y_train)

# 预测
predictions = model.predict(X_test)
```

## 2. FastText

### 原理

- 使用词向量和n-gram特征
- 层次softmax加速训练
- 支持子词信息

### 配置参数

```yaml
fasttext:
  dim: 100           # 词向量维度
  epoch: 25          # 训练轮数
  lr: 0.5            # 学习率
  wordNgrams: 2      # n-gram大小
```

### 使用示例

```python
from src.models.baseline.fasttext_model import FastTextClassifier

model = FastTextClassifier(dim=100, epoch=25)
model.train(train_texts, train_labels)

predictions, probas = model.predict(test_texts)
```

## 3. BERT

### 原理

- 基于Transformer的预训练语言模型
- 使用[CLS]向量进行分类
- 支持微调

### 配置参数

```yaml
bert:
  model_name: "hfl/chinese-bert-wwm-ext"
  num_labels: 10
  max_length: 128
  learning_rate: 2.0e-5
  epochs: 5
```

### 支持的中文BERT模型

- `hfl/chinese-bert-wwm-ext` (推荐)
- `bert-base-chinese`
- `hfl/chinese-roberta-wwm-ext`
- `hfl/chinese-macbert-base`

### 使用示例

```python
from src.models.bert.bert_classifier import BertClassifier
from src.models.bert.config import BertConfig

config = BertConfig(
    model_name="hfl/chinese-bert-wwm-ext",
    num_labels=10
)
model = BertClassifier(config=config)

# 训练使用BertTrainer
from src.training.bert_trainer import BertTrainer
trainer = BertTrainer(model)
trainer.train(train_dataloader, val_dataloader)
```

## 4. 模型优化

### 量化

将FP32权重转换为INT8，减少模型大小。

```python
from src.models.optimization.quantization import ModelQuantizer

quantizer = ModelQuantizer(dtype="int8")
quantized_model = quantizer.quantize(model)
```

### 剪枝

移除不重要的权重，提高稀疏性。

```python
from src.models.optimization.pruning import ModelPruner

pruner = ModelPruner(sparsity=0.5)
pruned_model = pruner.prune(model)
```

### 知识蒸馏

用大模型（教师）指导小模型（学生）训练。

```python
from src.models.optimization.distillation import DistillationTrainer

trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    temperature=4.0
)
