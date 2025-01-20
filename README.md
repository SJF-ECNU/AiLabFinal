# 多模态情感分析系统

这是一个基于BERT和ViT的多模态情感分析系统，能够同时处理文本和图像数据，进行三分类情感分析（消极、中性、积极）。

## 文件结构
注意，data.zip需要解压！！！解压为data/
```
.
├── config.py               # 配置文件
├── model.py               # 模型定义
├── train.py               # 训练脚本
├── dataset.py             # 数据集类定义
├── data_loader.py         # 数据加载器
├── main.py                # 主程序入口
├── predict.py             # 预测脚本
├── hyperparameter_search.py # 超参数搜索脚本
├── requirements.txt       # 依赖包列表
├── data/                  # 数据目录
│   ├── train.txt         # 训练集标签文件
│   ├── test_without_label.txt  # 测试集文件
│   ├── test_predictions.txt    # 预测结果文件
│   ├── *.txt             # 文本数据文件
│   └── *.jpg             # 图像数据文件
└── README.md             # 项目说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- 其他依赖见 requirements.txt

## 安装说明

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行说明

### 1. 数据准备
- 将训练数据放在 `data/` 目录下
- 确保数据格式符合要求（详见数据格式说明）

### 2. 参数配置
所有可配置的参数都在 `config.py` 文件中，主要包括：

1. 模型参数 (MODEL):
```python
MODEL = {
    'BERT_NAME': 'bert-base-uncased',  # BERT模型名称
    'VIT_NAME': 'google/vit-base-patch16-224',  # ViT模型名称
    'HIDDEN_DIM': 768,  # 隐藏层维度
    'NUM_CLASSES': 3,  # 分类类别数
    'DROPOUT_RATES': [0.1, 0.2, 0.3],  # Dropout率
    'USE_LAYER_NORM': True,  # 是否使用层归一化
}
```

2. 训练参数 (TRAINING):
```python
TRAINING = {
    'BATCH_SIZE': 32,  # 批次大小
    'NUM_EPOCHS': 20,  # 训练轮数
    'LEARNING_RATE': 2e-5,  # 学习率
    'WEIGHT_DECAY': 0.01,  # 权重衰减
    'WARMUP_RATIO': 0.1,  # 预热比例
    'EARLY_STOPPING': True,  # 是否启用早停
    'PATIENCE': 3,  # 早停耐心值
}
```

3. 数据增强参数 (DATA):
```python
DATA = {
    'TEXT_AUGMENT': True,  # 是否进行文本增强
    'IMAGE_AUGMENT': True,  # 是否进行图像增强
    'TEXT_REPLACE_RATIO': 0.1,  # 文本替换比例
    'IMAGE_SIZE': 224,  # 图像大小
}
```

### 3. 模型训练

#### 3.1 标准训练
```bash
python main.py
```

#### 3.2 超参数搜索
使用超参数搜索脚本来寻找最佳参数组合：
```bash
python hyperparameter_search.py --search_space search_space.json
```

超参数搜索配置示例 (search_space.json):
```json
{
    "TRAINING.LEARNING_RATE": [1e-5, 2e-5, 5e-5],
    "TRAINING.BATCH_SIZE": [16, 32, 64],
    "MODEL.DROPOUT_RATES": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
    "DATA.TEXT_REPLACE_RATIO": [0.1, 0.15, 0.2]
}
```

搜索结果将保存在 `grid_search_results_[timestamp].json` 文件中。

#### 3.3 消融实验
在 `main.py` 中取消注释相关代码块即可运行消融实验，包括：
- 仅文本模型
- 仅图像模型
- 完整多模态模型

### 4. 模型评估和预测
训练完成后：
- 最佳模型将保存为 `best_model_[timestamp].pth`
- 测试集预测结果将保存在 `data/test_predictions.txt`
- 训练过程指标可在 wandb 平台查看

### 5. 模型监控
本项目使用 Weights & Biases (wandb) 进行实验追踪，可以监控：
- 训练和验证损失
- 准确率和F1分数
- 各类别的性能指标
- 模态融合权重变化

在 `config.py` 中配置 wandb 参数：
```python
WANDB = {
    'PROJECT_NAME': 'multimodal-sentiment',
    'SAVE_CODE': True,
    'DISABLE_STATS': False
}
```

### 4. 模型预测

#### 4.1 使用预测脚本
使用训练好的模型对测试集进行预测：
```bash
python predict.py --model_path [model_path] --device [cuda/cpu]
```

参数说明：
- `--model_path`: 模型文件路径，默认为 'best_model_20250120_225501.pth'
- `--device`: 使用的设备，可选 'cuda' 或 'cpu'，默认自动选择

#### 4.2 预测结果
预测结果将保存在 `data/test_predictions.txt` 文件中，格式如下：
```
guid,tag
1,positive
2,negative
3,neutral
...
```

标签说明：
- positive: 积极情感
- neutral: 中性情感
- negative: 消极情感

#### 4.3 预测过程
1. 加载预训练模型
2. 读取测试集数据
3. 对每个样本进行预测
4. 生成预测报告，包括：
   - 总样本数
   - 成功预测数
   - 各类别的分布情况
5. 保存预测结果

## 数据格式说明

1. 训练集标签文件 (train.txt)：
```
guid,tag
1,positive
2,negative
3,neutral
...
```

2. 每个样本包含：
   - `[guid].txt`：文本文件
   - `[guid].jpg`：对应的图像文件

## 主要特性

- 多模态融合：结合BERT和ViT进行文本和图像的特征提取
- 交叉注意力机制：实现文本-图像特征的双向交互
- 动态特征融合：自适应调整不同模态特征的权重
- 数据增强：支持文本和图像的多种增强方式
- 训练优化：支持早停、学习率调度、梯度裁剪等

## 模型架构

1. 特征提取：
   - 文本：BERT-base
   - 图像：ViT-base

2. 特征融合：
   - 交叉注意力机制
   - 动态权重融合

3. 分类器：
   - 多层感知机
   - Focal Loss损失函数

## 参考文献

1. BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. ViT: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
3. Focal Loss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
4. Cross-Attention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
