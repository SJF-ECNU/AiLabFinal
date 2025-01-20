import torch
import numpy as np
import random
from model import MultiModalClassifier
from train import train_model
from data_loader import load_data
import wandb
from config import Config as cfg
import json
from datetime import datetime
from predict import predict_test_set

def set_seed(seed=42):
    """
    设置随机种子的函数
    
    输入:
        seed: int, 随机种子值
        
    输出:
        None
    
    功能:
        设置各个库的随机种子,确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_text_only_model(train_loader, val_loader, device):
    """
    训练仅文本模型的函数
    
    输入:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备(CPU/GPU)
        
    输出:
        float: 最佳验证集准确率
    
    功能:
        训练一个只使用文本特征的模型
    """
    print("\n=== 训练仅文本模型 ===")
    model = MultiModalClassifier()
    
    # 禁用图像特征
    def forward_hook(module, input, output):
        return torch.zeros_like(output)
    
    model.feature_extractor.image_transform.register_forward_hook(forward_hook)
    
    # 训练模型
    return train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device)

def train_image_only_model(train_loader, val_loader, device):
    """
    训练仅图像模型的函数
    
    输入:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备(CPU/GPU)
        
    输出:
        float: 最佳验证集准确率
    
    功能:
        训练一个只使用图像特征的模型
    """
    print("\n=== 训练仅图像模型 ===")
    model = MultiModalClassifier()
    
    # 禁用文本特征
    def forward_hook(module, input, output):
        return torch.zeros_like(output)
    
    model.feature_extractor.text_transform.register_forward_hook(forward_hook)
    
    # 训练模型
    return train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device)

def train_full_model(train_loader, val_loader, device):
    """
    训练完整多模态模型的函数
    
    输入:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备(CPU/GPU)
        
    输出:
        float: 最佳验证集准确率
    
    功能:
        训练完整的多模态模型
    """
    print("\n=== 训练完整多模态模型 ===")
    model = MultiModalClassifier()
    return train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device)

def main():
    """
    主函数
    
    输入:
        None
        
    输出:
        None
    
    功能:
        执行完整的训练流程,包括数据加载、模型训练和结果评估
    """
    # 设置随机种子
    set_seed(cfg.SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader = load_data(data_dir='data')
    
    if train_loader is None or val_loader is None:
        print("错误: 数据加载失败")
        return
    
    # 记录实验开始时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 进行消融实验
    results = {}
    text_model_path = f'best_text_model_{timestamp}.pth'
    image_model_path = f'best_image_model_{timestamp}.pth'

    
    # 为每个模型定义不同的保存路径
    full_model_path = f'best_full_model_{timestamp}.pth'
    
    # 训练完整多模态模型
    with wandb.init(project=cfg.WANDB.PROJECT_NAME, name=f"full_model_{timestamp}", reinit=True) as run:
        model = MultiModalClassifier()
        full_model_acc = train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device, full_model_path)
        results['full_model'] = full_model_acc
            
    # 1. 训练仅文本模型
    with wandb.init(project=cfg.WANDB.PROJECT_NAME, name=f"text_only_{timestamp}", reinit=True) as run:
        model = MultiModalClassifier()
        # 禁用图像特征
        def text_forward_hook(module, input, output):
            return torch.zeros_like(output)
        model.feature_extractor.image_transform.register_forward_hook(text_forward_hook)
        text_only_acc = train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device, text_model_path)
        results['text_only'] = text_only_acc
        
    # 2. 训练仅图像模型
    with wandb.init(project=cfg.WANDB.PROJECT_NAME, name=f"image_only_{timestamp}", reinit=True) as run:
        model = MultiModalClassifier()
        # 禁用文本特征
        def image_forward_hook(module, input, output):
            return torch.zeros_like(output)
        model.feature_extractor.text_transform.register_forward_hook(image_forward_hook)
        image_only_acc = train_model(model, train_loader, val_loader, cfg.TRAINING.NUM_EPOCHS, device, image_model_path)
        results['image_only'] = image_only_acc

    with open(f'ablation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'results': {
                'text_only': float(text_only_acc),
                'image_only': float(image_only_acc),
                'full_model': float(full_model_acc)
            }
        }, f, ensure_ascii=False, indent=2)
    
    # 打印结果比较
    print("\n=== 消融实验结果 ===")
    print(f"仅文本模型准确率: {text_only_acc:.4f}")
    print(f"仅图像模型准确率: {image_only_acc:.4f}")
    print(f"完整多模态模型准确率: {full_model_acc:.4f}")

    # 使用完整多模态模型进行预测
    predict_test_set(full_model_path, device)

if __name__ == '__main__':
    main() 