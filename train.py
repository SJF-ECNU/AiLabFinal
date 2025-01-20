import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import os
from config import Config as cfg
from datetime import datetime

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, num_epochs, device, model_save_path):
    """
    训练模型的主函数
    
    输入:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 训练设备(CPU/GPU)
        model_save_path: 模型保存路径
        
    输出:
        best_val_acc: float, 最佳验证集准确率
    
    功能:
        执行模型训练、验证和早停,记录训练过程,保存最佳模型
    """
    # 配置wandb
    os.environ['WANDB_SAVE_CODE'] = str(cfg.WANDB.SAVE_CODE).lower()
    
    # 初始化wandb
    run = wandb.init(
        project=cfg.WANDB.PROJECT_NAME,
        config={
            "learning_rate": cfg.TRAINING.LEARNING_RATE,
            "epochs": cfg.TRAINING.NUM_EPOCHS,
            "batch_size": cfg.TRAINING.BATCH_SIZE,
            "weight_decay": cfg.TRAINING.WEIGHT_DECAY,
            "warmup_ratio": cfg.TRAINING.WARMUP_RATIO,
            "model_type": "multimodal",
            "focal_loss_gamma": cfg.TRAINING.FOCAL_LOSS_GAMMA
        },
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=cfg.WANDB.DISABLE_STATS,
            _disable_meta=cfg.WANDB.DISABLE_META
        )
    )
    
    model = model.to(device)
    
    # 计算类别权重
    if cfg.TRAINING.USE_CLASS_WEIGHTS:
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['labels'].numpy())
        
        all_labels = np.array(all_labels)
        label_counts = np.bincount(all_labels)
        weights = len(all_labels) / (cfg.MODEL.NUM_CLASSES * label_counts)
    else:
        weights = None
    
    # 使用Focal Loss
    criterion = FocalLoss(
        alpha=torch.FloatTensor(weights) if weights is not None else None,
        gamma=cfg.TRAINING.FOCAL_LOSS_GAMMA
    )
    criterion = criterion.to(device)
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.TRAINING.LEARNING_RATE,
        weight_decay=cfg.TRAINING.WEIGHT_DECAY
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * cfg.TRAINING.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * cfg.TRAINING.WARMUP_RATIO),
        num_training_steps=num_training_steps
    )
    
    best_val_acc = 0.0
    no_improve = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = f'best_model/best_model_{timestamp}.pth'
    
    for epoch in range(cfg.TRAINING.NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_fusion_weights = []
        
        # 训练循环
        running_loss = 0.0
        running_preds = []
        running_labels = []
        step = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.TRAINING.NUM_EPOCHS}'):
            optimizer.zero_grad()
            
            # 准备输入
            text_dict = {
                'input_ids': batch['text']['input_ids'].to(device),
                'attention_mask': batch['text']['attention_mask'].to(device)
            }
            image_inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(text_dict, image_inputs)
            
            # 获取特征融合权重
            fusion_weights = model.feature_extractor.fusion_weights
            if fusion_weights is not None:
                all_fusion_weights.append(fusion_weights.detach().cpu().mean(0))
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if cfg.TRAINING.GRADIENT_CLIP_VAL > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.TRAINING.GRADIENT_CLIP_VAL
                )
            
            optimizer.step()
            scheduler.step()
            
            # 累积损失和预测结果
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_preds.extend(preds.cpu().numpy())
            running_labels.extend(labels.cpu().numpy())
            
            # 每100个batch记录一次训练指标
            step += 1
            if step % 100 == 0:
                # 计算当前batch的指标
                batch_metrics = calculate_metrics(running_labels, running_preds)
                avg_loss = running_loss / step
                
                # 记录到wandb
                wandb.log({
                    "train/step": epoch * len(train_loader) + step,
                    "train/batch_loss": avg_loss,
                    "train/batch_accuracy": batch_metrics['accuracy'],
                    "train/batch_macro_f1": batch_metrics['macro_f1'],
                    "train/batch_negative_acc": batch_metrics['class_accuracies']['Negative'],
                    "train/batch_neutral_acc": batch_metrics['class_accuracies']['Neutral'],
                    "train/batch_positive_acc": batch_metrics['class_accuracies']['Positive'],
                    "train/learning_rate": scheduler.get_last_lr()[0]
                })
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算训练指标
        train_metrics = calculate_metrics(all_labels, all_preds)
        epoch_loss = total_loss / len(train_loader)
        
        # 计算平均融合权重
        if all_fusion_weights:
            avg_fusion_weights = torch.stack(all_fusion_weights).mean(0)
            
            # 记录epoch级别的训练指标到wandb
            metrics_dict = {
                "epoch": epoch + 1,
                "train/epoch_loss": epoch_loss,
                "train/epoch_accuracy": train_metrics['accuracy'],
                "train/epoch_macro_f1": train_metrics['macro_f1'],
                "train/epoch_negative_acc": train_metrics['class_accuracies']['Negative'],
                "train/epoch_neutral_acc": train_metrics['class_accuracies']['Neutral'],
                "train/epoch_positive_acc": train_metrics['class_accuracies']['Positive'],
                "train/fusion_weight_text": avg_fusion_weights[0].item(),
                "train/fusion_weight_image": avg_fusion_weights[1].item(),
                "train/fusion_weight_text2image": avg_fusion_weights[2].item(),
                "train/fusion_weight_image2text": avg_fusion_weights[3].item()
            }
            wandb.log(metrics_dict)
        
        # 打印训练信息
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {epoch_loss:.4f}')
        print_metrics(train_metrics, "Training")
        
        # 验证
        if (epoch + 1) % cfg.TRAINING.VAL_INTERVAL == 0 and epoch>=4:
            val_metrics = evaluate_model(model, val_loader, device, criterion)
            print_metrics(val_metrics, "Validation")
            
            # 记录验证指标到wandb
            wandb.log({
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/macro_f1": val_metrics['macro_f1'],
                "val/negative_acc": val_metrics['class_accuracies']['Negative'],
                "val/neutral_acc": val_metrics['class_accuracies']['Neutral'],
                "val/positive_acc": val_metrics['class_accuracies']['Positive'],
                "epoch": epoch + 1
            })
            
            # 早停检查
            current_val_acc = val_metrics['accuracy']
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                # 保存最佳模型
                print(f"\n保存最佳模型到 {model_save_path}，验证集准确率: {current_val_acc:.4f}")
                torch.save(model.state_dict(), model_save_path)
                no_improve = 0
            else:
                no_improve += 1
                if cfg.TRAINING.EARLY_STOPPING and no_improve >= cfg.TRAINING.PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
    
    # 不要删除最佳模型文件
    wandb.finish()
    return best_val_acc

def evaluate_model(model, val_loader, device, criterion):
    """
    评估模型性能的函数
    
    输入:
        model: 待评估的模型
        val_loader: 验证数据加载器
        device: 评估设备(CPU/GPU)
        criterion: 损失函数
        
    输出:
        metrics: dict, 包含各项评估指标的字典
    
    功能:
        在验证集上评估模型性能,计算损失和各项指标
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 准备输入
            text_dict = {
                'input_ids': batch['text']['input_ids'].to(device),
                'attention_mask': batch['text']['attention_mask'].to(device)
            }
            image_inputs = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(text_dict, image_inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 收集预测结果
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def calculate_metrics(true_labels, predictions):
    """
    计算评估指标的函数
    
    输入:
        true_labels: 真实标签列表
        predictions: 预测标签列表
        
    输出:
        dict: 包含准确率、F1分数和各类别准确率的字典
    
    功能:
        计算模型预测的各项评估指标
    """
    # 计算整体指标
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    report = classification_report(true_labels, predictions,
                                 target_names=['Negative', 'Neutral', 'Positive'],
                                 output_dict=True)
    
    # 计算每个类别的准确率
    class_accuracies = {
        'Negative': report['Negative']['recall'],
        'Neutral': report['Neutral']['recall'],
        'Positive': report['Positive']['recall']
    }
    
    return {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'class_accuracies': class_accuracies
    }

def print_metrics(metrics, phase=""):
    """
    打印评估指标的函数
    
    输入:
        metrics: 包含各项指标的字典
        phase: 当前阶段名称(训练/验证)
        
    输出:
        None
    
    功能:
        格式化打印评估指标
    """
    print(f"\n{phase} Metrics:")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClass-wise Accuracies:")
    for class_name, acc in metrics['class_accuracies'].items():
        print(f"{class_name}: {acc:.4f}") 