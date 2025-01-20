import os
import re
from collections import Counter
import numpy as np

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

def get_sentiment(text):
    """
    简单情感分析函数
    
    输入:
        text: str, 输入文本
        
    输出:
        str: 情感标签('positive'/'negative'/'neutral')
        
    功能:
        基于关键词的简单情感分析
    """
    positive_words = ['happy', 'great', 'good', 'nice', 'love', '😊', '😄', 'awesome', 'excellent', 'wonderful', 'best', 'amazing']
    negative_words = ['sad', 'bad', 'hate', 'terrible', 'awful', '😢', '😠', 'worst', 'horrible', 'poor', 'disappointed']
    
    text = text.lower()
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    return 'neutral'

def analyze_data(data_dir='data'):
    """
    数据分析函数
    
    输入:
        data_dir: str, 数据目录路径
        
    输出:
        None
        
    功能:
        分析数据集的文本和图像特征,生成统计报告
    """
    # 获取所有文件
    files = os.listdir(data_dir)
    txt_files = sorted([f for f in files if f.endswith('.txt')])
    jpg_files = sorted([f for f in files if f.endswith('.jpg')])
    
    # 基本统计
    print("\n=== 基本统计 ===")
    print(f"文本文件总数: {len(txt_files)}")
    print(f"图像文件总数: {len(jpg_files)}")
    
    # 文本分析
    text_lengths = []
    word_counts = []
    all_words = []
    sentiments = []
    emoji_count = 0
    url_count = 0
    mention_count = 0
    retweet_count = 0
    
    for txt_file in txt_files:
        with open(os.path.join(data_dir, txt_file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            
            # 统计特殊元素
            emoji_count += len(re.findall(r'[\U0001F300-\U0001F9FF]', text))
            url_count += len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
            mention_count += len(re.findall(r'@\w+', text))
            if text.startswith('RT '):
                retweet_count += 1
            
            cleaned_text = clean_tweet(text)
            
            # 情感分析
            sentiment = get_sentiment(cleaned_text)
            sentiments.append(sentiment)
            
            # 统计文本长度
            text_lengths.append(len(cleaned_text))
            
            # 统计词数
            words = cleaned_text.split()
            word_counts.append(len(words))
            all_words.extend(words)
    
    # 打印统计结果
    print("\n=== 文本特征统计 ===")
    print(f"Emoji数量: {emoji_count}")
    print(f"URL链接数量: {url_count}")
    print(f"@提及数量: {mention_count}")
    print(f"转发数量: {retweet_count}")
    
    print("\n=== 文本长度统计 ===")
    print(f"平均文本长度: {np.mean(text_lengths):.2f} 字符")
    print(f"最短文本长度: {min(text_lengths)} 字符")
    print(f"最长文本长度: {max(text_lengths)} 字符")
    print(f"文本长度标准差: {np.std(text_lengths):.2f}")
    
    print("\n=== 词数统计 ===")
    print(f"平均词数: {np.mean(word_counts):.2f}")
    print(f"最少词数: {min(word_counts)}")
    print(f"最多词数: {max(word_counts)}")
    print(f"词数标准差: {np.std(word_counts):.2f}")
    
    print("\n=== 情感分布 ===")
    sentiment_counter = Counter(sentiments)
    total = len(sentiments)
    for sentiment, count in sentiment_counter.items():
        percentage = (count / total) * 100
        print(f"{sentiment}: {count} ({percentage:.2f}%)")
    
    print("\n=== 高频词统计（top 20）===")
    word_freq = Counter(all_words)
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")
    
    # 图像文件大小统计
    image_sizes = []
    for jpg_file in jpg_files:
        size = os.path.getsize(os.path.join(data_dir, jpg_file)) / 1024  # 转换为KB
        image_sizes.append(size)
    
    print("\n=== 图像统计 ===")
    print(f"平均图像大小: {np.mean(image_sizes):.2f} KB")
    print(f"最小图像大小: {min(image_sizes):.2f} KB")
    print(f"最大图像大小: {max(image_sizes):.2f} KB")
    print(f"图像大小标准差: {np.std(image_sizes):.2f} KB")
    
    # 打印一些示例
    print("\n=== 文本示例 ===")
    for i in range(min(5, len(txt_files))):
        with open(os.path.join(data_dir, txt_files[i]), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
            cleaned = clean_tweet(text)
            sentiment = get_sentiment(cleaned)
            print(f"\n示例 {i+1}:")
            print(f"原始文本: {text}")
            print(f"清理后文本: {cleaned}")
            print(f"情感标签: {sentiment}")

if __name__ == '__main__':
    analyze_data() 