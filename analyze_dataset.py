import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import re
import sys

def clean_tweet(tweet):
    """
    清理推文文本的函数
    
    输入:
        tweet: str, 原始推文文本
        
    输出:
        str: 清理后的文本
        
    功能:
        移除URL、@用户名等无关内容
    """
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def analyze_dataset(data_dir='data'):
    """
    数据集分析函数
    
    输入:
        data_dir: str, 数据目录路径
        
    输出:
        None
        
    功能:
        分析数据集的各项统计特征并打印报告
    """
    try:
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            print(f"错误：数据目录 '{data_dir}' 不存在")
            return
        
        # 检查标签文件是否存在
        label_file = os.path.join(data_dir, 'train.txt')
        if not os.path.exists(label_file):
            print(f"错误：标签文件 '{label_file}' 不存在")
            return
        
        # 读取标签文件
        try:
            label_df = pd.read_csv(label_file)
        except Exception as e:
            print(f"错误：无法读取标签文件：{str(e)}")
            return
        
        # 1. 标签分布分析
        print("\n=== 标签分布 ===")
        label_counts = label_df['tag'].value_counts()
        total_samples = len(label_df)
        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            print(f"{label}: {count} ({percentage:.2f}%)")
        
        # 获取文本和图像文件
        files = os.listdir(data_dir)
        txt_files = sorted([f for f in files if f.endswith('.txt') and f != 'train.txt'])
        jpg_files = sorted([f for f in files if f.endswith('.jpg')])
        
        # 2. 文本分析
        print("\n=== 文本分析 ===")
        text_lengths = []
        word_counts = []
        all_words = []
        emoji_count = 0
        url_count = 0
        mention_count = 0
        hashtag_count = 0
        
        valid_guids = set(label_df['guid'].values)
        
        for txt_file in txt_files:
            try:
                base_name = txt_file[:-4]
                if int(base_name) in valid_guids:
                    with open(os.path.join(data_dir, txt_file), 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                        
                        # 统计特殊元素（在清理文本之前）
                        emoji_count += len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
                        url_count += len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
                        mention_count += len(re.findall(r'@\w+', text))
                        hashtag_count += len(re.findall(r'#\w+', text))
                        
                        # 清理文本后的统计
                        cleaned_text = clean_tweet(text)
                        if cleaned_text:  # 只统计非空文本
                            text_lengths.append(len(cleaned_text))
                            words = cleaned_text.split()
                            if words:  # 只统计有词的文本
                                word_counts.append(len(words))
                                all_words.extend(words)
            except Exception as e:
                print(f"警告：处理文件 {txt_file} 时出错：{str(e)}")
                continue
        
        if text_lengths:
            print(f"文本数量: {len(text_lengths)}")
            print(f"\n文本长度统计:")
            print(f"- 平均长度: {np.mean(text_lengths):.2f} 字符")
            print(f"- 最短长度: {min(text_lengths)} 字符")
            print(f"- 最长长度: {max(text_lengths)} 字符")
            print(f"- 长度标准差: {np.std(text_lengths):.2f}")
        
        if word_counts:
            print(f"\n词数统计:")
            print(f"- 平均词数: {np.mean(word_counts):.2f}")
            print(f"- 最少词数: {min(word_counts)}")
            print(f"- 最多词数: {max(word_counts)}")
            print(f"- 词数标准差: {np.std(word_counts):.2f}")
        
        print(f"\n特殊元素统计:")
        print(f"- Emoji数量: {emoji_count}")
        print(f"- URL链接数量: {url_count}")
        print(f"- @提及数量: {mention_count}")
        print(f"- #话题标签数量: {hashtag_count}")
        
        if all_words:
            print("\n=== 高频词统计（top 20）===")
            word_freq = Counter(all_words)
            for word, count in word_freq.most_common(20):
                print(f"{word}: {count}")
        
        # 3. 图像分析
        print("\n=== 图像分析 ===")
        image_sizes = []
        image_dimensions = []
        
        for jpg_file in jpg_files:
            try:
                base_name = jpg_file[:-4]
                if int(base_name) in valid_guids:
                    # 文件大小
                    size = os.path.getsize(os.path.join(data_dir, jpg_file)) / 1024  # KB
                    image_sizes.append(size)
                    
                    # 图像尺寸
                    with Image.open(os.path.join(data_dir, jpg_file)) as img:
                        width, height = img.size
                        image_dimensions.append((width, height))
            except Exception as e:
                print(f"警告：处理图像 {jpg_file} 时出错：{str(e)}")
                continue
        
        if image_sizes:
            print(f"图像数量: {len(image_sizes)}")
            print(f"\n图像大小统计:")
            print(f"- 平均大小: {np.mean(image_sizes):.2f} KB")
            print(f"- 最小大小: {min(image_sizes):.2f} KB")
            print(f"- 最大大小: {max(image_sizes):.2f} KB")
            print(f"- 大小标准差: {np.std(image_sizes):.2f} KB")
        
        if image_dimensions:
            widths, heights = zip(*image_dimensions)
            print(f"\n图像尺寸统计:")
            print(f"- 平均宽度: {np.mean(widths):.2f} 像素")
            print(f"- 平均高度: {np.mean(heights):.2f} 像素")
            print(f"- 宽度范围: {min(widths)} - {max(widths)} 像素")
            print(f"- 高度范围: {min(heights)} - {max(heights)} 像素")
        
        # 4. 每个类别的示例
        print("\n=== 每个类别的示例 ===")
        for label in ['negative', 'neutral', 'positive']:
            print(f"\n{label.upper()} 类别示例:")
            try:
                sample_guids = label_df[label_df['tag'] == label]['guid'].sample(min(3, len(label_df[label_df['tag'] == label]))).values
                for guid in sample_guids:
                    txt_file = f"{guid}.txt"
                    file_path = os.path.join(data_dir, txt_file)
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read().strip()
                                print(f"ID: {guid}")
                                print(f"原始文本: {text}")
                                print(f"清理后文本: {clean_tweet(text)}\n")
                        except Exception as e:
                            print(f"警告：读取文件 {txt_file} 时出错：{str(e)}")
            except Exception as e:
                print(f"警告：处理 {label} 类别示例时出错：{str(e)}")
                continue
                
    except Exception as e:
        print(f"错误：分析过程中出现异常：{str(e)}")
        return

if __name__ == '__main__':
    analyze_dataset() 