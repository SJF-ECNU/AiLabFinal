import torch
from model import MultiModalClassifier
from data_loader import load_data
from tqdm import tqdm
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def predict_test_set(model_path, device=None):
    """
    对测试集进行预测的函数
    
    输入:
        model_path: str, 模型文件路径
        device: torch.device, 运行设备(CPU/GPU)
        
    输出:
        None, 预测结果保存到文件
        
    功能:
        加载模型并对测试集进行预测,保存预测结果
    """
    print("\n=== 预测测试集 ===")
    
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载最佳模型
    print(f"加载模型: {model_path}")
    model = MultiModalClassifier()
    # 使用weights_only=True来安全加载模型
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # 标签映射
    label_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    
    # 加载测试数据
    test_loader = load_data(data_dir='data', split='test')
    if test_loader is None:
        print("错误: 测试数据加载失败")
        return
    
    predictions = []
    guids = []
    
    print("\n开始预测...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='预测测试集'):
            # 准备输入
            text_dict = {
                'input_ids': batch['text']['input_ids'].to(device),
                'attention_mask': batch['text']['attention_mask'].to(device)
            }
            image_inputs = batch['image'].to(device)
            
            # 获取预测结果
            outputs = model(text_dict, image_inputs)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            guids.extend(batch['guid'])
    
    print("\n保存预测结果...")
    # 读取原始文件内容
    with open('test_without_label.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 创建guid到预测标签的映射
    pred_map = {}
    for guid, pred in zip(guids, predictions):
        pred_map[str(int(guid))] = label_map[int(pred)]  # 转换为文本标签
    
    # 更新文件内容
    new_lines = []
    header = True
    updated_count = 0
    total_count = 0
    
    for line in lines:
        if header:
            new_lines.append(line)
            header = False
            continue
            
        total_count += 1
        parts = line.strip().split(',')
        guid = parts[0]
        
        if guid in pred_map:
            new_lines.append(f"{guid},{pred_map[guid]}\n")
            updated_count += 1
        else:
            new_lines.append(line)
    
    # 保存预测结果
    output_path = 'test_predictions.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\n预测完成:")
    print(f"- 总样本数: {total_count}")
    print(f"- 成功预测: {updated_count}")
    print(f"- 预测结果已保存到: {output_path}")
    
    # 打印预测标签的分布
    pred_counts = np.bincount(predictions)
    print("\n预测结果分布:")
    for i, count in enumerate(pred_counts):
        percentage = count / len(predictions) * 100
        print(f"- {label_map[i]}: {count} ({percentage:.2f}%)")
    
    # 打印一些预测示例
    print("\n预测示例:")
    for i, (guid, pred) in enumerate(zip(guids[:5], predictions[:5])):
        print(f"GUID: {int(guid)}, 预测标签: {label_map[int(pred)]}")
        # 验证映射是否正确
        print(f"映射验证 - GUID {int(guid)}: {pred_map[str(int(guid))]}")

def main():
    """
    主函数
    
    输入:
        None
        
    输出:
        None
        
    功能:
        解析命令行参数并执行预测流程
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用训练好的模型对测试集进行预测')
    parser.add_argument('--model_path', type=str, 
                      default='best_model_20250120_225501.pth',
                      help='模型文件路径')
    parser.add_argument('--device', type=str, 
                      choices=['cuda', 'cpu'], 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='使用的设备 (默认: cuda if available else cpu)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    
    # 执行预测
    predict_test_set(args.model_path, device)

if __name__ == '__main__':
    main() 