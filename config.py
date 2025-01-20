class ModelConfig:
    # 模型架构参数
    TEXT_DIM = 768  # BERT hidden size
    IMAGE_DIM = 768  # ViT hidden size
    HIDDEN_DIM = 512  # 隐藏层维度
    FUSION_DIM = 2048  # 特征融合后的维度
    NUM_CLASSES = 3  # 类别数
    DROPOUT_RATES = [0.3, 0.2, 0.1]  # 不同层的dropout率

    # 特征提取器参数
    BERT_NAME = 'bert-base-uncased'
    VIT_NAME = 'google/vit-base-patch16-224'
    
    # 注意力机制参数
    ATTENTION_DROPOUT = 0.1
    ATTENTION_HIDDEN_DIM = 512
    
    # 分类器参数
    CLASSIFIER_DIMS = [2048, 1024, 512, 256]  # 分类器各层维度
    USE_LAYER_NORM = True  # 是否使用层归一化
    
class TrainingConfig:
    # 训练参数
    BATCH_SIZE = 16
    NUM_EPOCHS = 12
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    GRADIENT_CLIP_VAL = 1.0
    
    # 早停参数
    PATIENCE = 3
    EARLY_STOPPING = True
    
    # 损失函数参数
    FOCAL_LOSS_GAMMA = 2.0
    USE_CLASS_WEIGHTS = True
    
    # 验证参数
    VAL_INTERVAL = 1  # 每多少个epoch验证一次
    TRAIN_VAL_SPLIT = 0.8  # 训练集验证集分割比例
    
class DataConfig:
    # 数据处理参数
    MAX_TEXT_LENGTH = 128
    IMAGE_SIZE = 224
    NUM_WORKERS = 8
    
    # 数据增强参数
    TEXT_AUGMENT = False  # 默认不使用文本增强
    IMAGE_AUGMENT = False  # 默认不使用图像增强
    TEXT_REPLACE_RATIO = 0.2  # 文本替换比例
    ROTATION_DEGREES = 15  # 图像旋转角度
    COLOR_JITTER = {  # 图像颜色增强参数
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    
    # 数据平衡参数
    USE_WEIGHTED_SAMPLER = True
    USE_DATA_AUGMENTATION = False  # 默认不使用数据增强
    
class WandbConfig:
    # Wandb配置
    PROJECT_NAME = "multimodal-sentiment"
    SAVE_CODE = False
    DISABLE_STATS = True
    DISABLE_META = True
    LOG_INTERVAL = 100  # 每多少个batch记录一次
    
    # 记录的指标
    METRICS = [
        'loss',
        'accuracy',
        'macro_f1',
        'class_accuracies',
        'fusion_weights'
    ]

class Config:
    MODEL = ModelConfig
    TRAINING = TrainingConfig
    DATA = DataConfig
    WANDB = WandbConfig
    
    # 其他全局配置
    SEED = 42
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    DATA_DIR = 'data'
    
    @classmethod
    def print_config(cls):
        """打印所有配置信息"""
        print("\n=== 配置信息 ===")
        for category in [cls.MODEL, cls.TRAINING, cls.DATA, cls.WANDB]:
            print(f"\n{category.__name__}:")
            for key, value in category.__dict__.items():
                if not key.startswith('_'):  # 不打印私有属性
                    print(f"  {key}: {value}") 