import os
import json
import re
import pandas as pd
import numpy as np
from dataset import create_data_loaders
from config import Config as cfg

def clean_tweet(tweet):
    """
    清理推文文本
    
    输入:
        tweet: str, 原始推文文本
        
    输出:
        str: 清理后的文本
        
    功能:
        移除URL、@用户名等无关内容
    """
    # 移除URL
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # 移除@用户名
    tweet = re.sub(r'@\w+', '', tweet)
    # 移除RT标记
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # 移除多余的空白字符
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def load_data(data_dir='data', split='train'):
    """
    加载数据集
    
    输入:
        data_dir: str, 数据目录路径
        split: str, 数据集类型('train'或'test')
        
    输出:
        tuple/DataLoader: 训练集和验证集的DataLoader,或测试集DataLoader
        
    功能:
        加载和预处理数据集
    """
    try:
        if split == 'test':
            # 修改测试集文件路径
            test_file = 'test_without_label.txt'  # 从data_dir中移出
            df = pd.read_csv(test_file)
            
            # 将guid列转换为整数
            df['guid'] = df['guid'].astype(int)
            print("GUID类型:", df['guid'].dtype)
            print("GUID示例:", df['guid'].iloc[0])
            
            # 准备数据
            text_data = []
            image_paths = []
            guids = []
            
            print(f"\n加载测试集数据...")
            print(f"测试集文件路径: {test_file}")
            print(f"测试集样本数量: {len(df)}")
            
            # 读取数据
            missing_files = {'text': [], 'image': []}
            for _, row in df.iterrows():
                guid = int(row['guid'])  # 确保是整数
                
                # 构建文件路径
                text_file = os.path.join(data_dir, str(guid) + '.txt')  # 使用str()转换
                image_file = os.path.join(data_dir, str(guid) + '.jpg')  # 使用str()转换
                
                print(f"尝试读取文件: {text_file}")  # 调试信息
                
                # 检查文件是否存在
                if not os.path.exists(text_file):
                    missing_files['text'].append(int(guid))  # 存储整数
                    continue
                    
                if not os.path.exists(image_file):
                    missing_files['image'].append(int(guid))  # 存储整数
                    continue
                
                # 读取文本（尝试多种编码）
                text = None
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                for encoding in encodings:
                    try:
                        with open(text_file, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                            text = clean_tweet(text)  # 清理文本
                        break
                    except Exception as e:
                        if encoding == encodings[-1]:  # 如果是最后一种编码也失败了
                            print(f"警告: 无法读取文本文件 {text_file}")
                            text = None
                        continue
                
                if text is None:
                    continue
                    
                text_data.append(text)
                image_paths.append(image_file)
                guids.append(int(guid))  # 存储整数
            
            if len(text_data) == 0:
                if missing_files['text'] or missing_files['image']:
                    print("\n缺失文件统计:")
                    if missing_files['text']:
                        print(f"缺失文本文件数量: {len(missing_files['text'])}")
                        print("缺失的文本文件:", [str(g) for g in missing_files['text'][:10]], "..." if len(missing_files['text']) > 10 else "")
                    if missing_files['image']:
                        print(f"缺失图像文件数量: {len(missing_files['image'])}")
                        print("缺失的图像文件:", [str(g) for g in missing_files['image'][:10]], "..." if len(missing_files['image']) > 10 else "")
                raise Exception("没有找到有效的测试数据")
            
            print(f"成功加载测试集数据: {len(text_data)} 个样本")
            
            # 创建测试集数据加载器
            return create_data_loaders(
                text_data, image_paths, guids,  # 测试集数据
                None, None, None,  # 无验证集
                is_test=True  # 标记为测试集模式
            )
        
        else:  # split == 'train'
            # 修改训练集标签文件路径
            label_file = 'train.txt'  # 从data_dir中移出
            df = pd.read_csv(label_file)
            
            # 将guid列转换为整数
            df['guid'] = df['guid'].astype(int)
            
            # 标签映射
            label_map = {
                'negative': 0,
                'neutral': 1,
                'positive': 2
            }
            
            # 准备数据
            text_data = []
            image_paths = []
            labels = []
            
            # 统计标签分布
            label_counts = df['tag'].value_counts()
            total_samples = len(df)
            
            print("\nData Statistics:")
            print(f"Total samples: {total_samples}")
            print("Label distribution:")
            for label, count in label_counts.items():
                percentage = count / total_samples * 100
                print(f"- {label}: {count} ({percentage:.2f}%)")
            
            # 读取数据
            for _, row in df.iterrows():
                guid = row['guid']  # 获取整数guid
                label = label_map[row['tag'].lower()]  # 将文本标签转换为数字
                
                # 构建文件路径
                text_file = os.path.join(data_dir, f"{guid}.txt")
                image_file = os.path.join(data_dir, f"{guid}.jpg")
                
                # 检查文件是否存在
                if not os.path.exists(text_file):
                    print(f"警告: 文本文件不存在 - {text_file}")
                    continue
                    
                if not os.path.exists(image_file):
                    print(f"警告: 图像文件不存在 - {image_file}")
                    continue
                
                # 读取文本（尝试多种编码）
                text = None
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                for encoding in encodings:
                    try:
                        with open(text_file, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                            text = clean_tweet(text)  # 清理文本
                        break
                    except Exception as e:
                        if encoding == encodings[-1]:  # 如果是最后一种编码也失败了
                            print(f"警告: 无法读取文本文件 {text_file}")
                            text = None
                        continue
                
                if text is None:
                    continue
                    
                text_data.append(text)
                image_paths.append(image_file)
                labels.append(label)
            
            if len(text_data) == 0:
                raise Exception("没有找到有效的训练数据")
            
            # 打印一些示例
            print("\nSample examples:\n")
            for i in range(min(3, len(text_data))):
                print(f"Example {i+1}:")
                print(f"Text: {text_data[i]}")
                print(f"Image path: {image_paths[i]}")
                print(f"Label: {labels[i]} ({list(label_map.keys())[list(label_map.values()).index(labels[i])]})\n")
            
            # 先进行训练集和验证集的分割
            split_idx = int(len(labels) * cfg.TRAINING.TRAIN_VAL_SPLIT)
            indices = np.random.permutation(len(labels))
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # 分别获取训练集和验证集的数据
            train_text = [text_data[i] for i in train_indices]
            train_images = [image_paths[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]
            
            val_text = [text_data[i] for i in val_indices]
            val_images = [image_paths[i] for i in val_indices]
            val_labels = [labels[i] for i in val_indices]
            
            print(f"\n成功加载数据:")
            print(f"训练集大小: {len(train_text)}")
            print(f"验证集大小: {len(val_text)}")
            
            # 创建数据加载器（只对训练集进行增强）
            return create_data_loaders(
                train_text, train_images, train_labels,  # 训练集数据
                val_text, val_images, val_labels  # 验证集数据
            )
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_num_samples(data_dir='data'):
    """
    获取数据集样本数量
    
    输入:
        data_dir: str, 数据目录路径
        
    输出:
        int: 数据集中的样本数量
        
    功能:
        统计数据集中的样本总数
    """
    # 修改训练集标签文件路径
    label_df = pd.read_csv('train.txt')  # 从data_dir中移出
    return len(label_df) 