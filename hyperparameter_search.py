import torch
import numpy as np
import random
from itertools import product
from model import MultiModalClassifier
from train import train_model
from data_loader import load_data
from config import Config as cfg
import wandb
from datetime import datetime
import json

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

def update_config(param_dict):
    """
    更新配置参数的函数
    
    输入:
        param_dict: dict, 新的参数配置
        
    输出:
        None
        
    功能:
        根据输入更新全局配置参数
    """
    if 'learning_rate' in param_dict:
        cfg.TRAINING.LEARNING_RATE = param_dict['learning_rate']
    if 'batch_size' in param_dict:
        cfg.TRAINING.BATCH_SIZE = param_dict['batch_size']
    if 'dropout_rates' in param_dict:
        cfg.MODEL.DROPOUT_RATES = param_dict['dropout_rates']
    if 'hidden_dim' in param_dict:
        cfg.MODEL.HIDDEN_DIM = param_dict['hidden_dim']
        # 同时更新相关的维度参数
        cfg.MODEL.FUSION_DIM = param_dict['hidden_dim'] * 4
        cfg.MODEL.CLASSIFIER_DIMS = [
            param_dict['hidden_dim'] * 4,
            param_dict['hidden_dim'] * 2,
            param_dict['hidden_dim'],
            param_dict['hidden_dim'] // 2
        ]
    if 'focal_loss_gamma' in param_dict:
        cfg.TRAINING.FOCAL_LOSS_GAMMA = param_dict['focal_loss_gamma']

def grid_search():
    """
    网格搜索函数
    
    输入:
        None
        
    输出:
        None
        
    功能:
        执行超参数网格搜索,找到最优参数组合并保存结果
    """
    # 设置随机种子
    set_seed(cfg.SEED)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（只加载一次）
    print("加载数据...")
    train_loader, val_loader = load_data(data_dir='data')
    
    if train_loader is None or val_loader is None:
        print("错误: 数据加载失败")
        return
    
    # 定义超参数搜索空间
    param_grid = {
        'learning_rate': [1e-5, 2e-5],
        'batch_size': [16, 32],
        'dropout_rates': [[0.2, 0.2, 0.2], [0.3, 0.2, 0.1]],
        'hidden_dim': [256, 512],
        'focal_loss_gamma': [1.5, 2.0]
    }
    
    # 生成所有参数组合
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in product(*param_grid.values())]
    
    # 创建结果记录
    results = []
    best_f1 = 0
    best_params = None
    
    # 记录实验开始时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n开始网格搜索，共 {len(param_combinations)} 组参数组合")
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\n正在测试第 {i}/{len(param_combinations)} 组参数:")
        for k, v in params.items():
            print(f"{k}: {v}")
        
        # 更新配置
        update_config(params)
        
        # 初始化新的wandb运行
        run = wandb.init(
            project=cfg.WANDB.PROJECT_NAME,
            config=params,
            name=f"grid_search_{i}_{timestamp}",
            reinit=True
        )
        
        try:
            # 训练模型
            model = MultiModalClassifier()
            model = model.to(device)
            current_f1 = train_model(model, train_loader, val_loader, 
                                   cfg.TRAINING.NUM_EPOCHS, device)
            
            # 记录结果
            result = {
                'params': params,
                'macro_f1': current_f1
            }
            results.append(result)
            
            # 更新最佳结果
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = params.copy()
                
            print(f"当前组合的Macro F1: {current_f1:.4f}")
            
        except Exception as e:
            print(f"训练出错: {str(e)}")
            continue
        finally:
            wandb.finish()
    
    # 保存所有结果
    with open(f'grid_search_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump({
            'all_results': results,
            'best_result': {
                'params': best_params,
                'macro_f1': best_f1
            }
        }, f, ensure_ascii=False, indent=2)
    
    print("\n网格搜索完成!")
    print(f"最佳Macro F1: {best_f1:.4f}")
    print("最佳参数组合:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    grid_search() 