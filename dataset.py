import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
import numpy as np
from torch.utils.data import WeightedRandomSampler
import nltk
from nltk.corpus import wordnet
import random
import torchvision.transforms.functional as TF
from config import Config as cfg

class MultiModalDataset(Dataset):
    """
    多模态数据集类
    
    输入:
        text_data: 文本数据列表
        image_paths: 图像路径列表
        labels: 标签列表
        max_length: BERT分词器最大长度
        augment: 是否进行数据增强
        is_test: 是否为测试集
        
    输出:
        None
        
    功能:
        加载和预处理多模态数据
    """
    def __init__(self, text_data, image_paths, labels, max_length=None, augment=False, is_test=False):
        """
        初始化数据集
        Args:
            text_data: 文本数据列表
            image_paths: 图像路径列表
            labels: 标签列表（测试集时为guid列表）
            max_length: BERT分词器的最大长度
            augment: 是否进行数据增强
            is_test: 是否为测试集模式
        """
        self.text_data = text_data
        self.image_paths = image_paths
        self.labels = labels
        self.max_length = max_length or cfg.DATA.MAX_TEXT_LENGTH
        self.augment = augment
        self.is_test = is_test
        
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
        
        # 下载nltk数据（如果需要）
        if cfg.DATA.TEXT_AUGMENT and not is_test:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                nltk.download('averaged_perceptron_tagger')
        
        # 基础图像预处理
        self.basic_transform = transforms.Compose([
            transforms.Resize((cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 增强图像预处理
        self.augment_transform = transforms.Compose([
            transforms.Resize((cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE)),
            transforms.RandomRotation(cfg.DATA.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=cfg.DATA.COLOR_JITTER['brightness'],
                contrast=cfg.DATA.COLOR_JITTER['contrast'],
                saturation=cfg.DATA.COLOR_JITTER['saturation'],
                hue=cfg.DATA.COLOR_JITTER['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def get_synonyms(self, word):
        """
        获取单词同义词的函数
        
        输入:
            word: str, 输入单词
            
        输出:
            list: 同义词列表
            
        功能:
            查找单词的同义词
        """
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.append(lemma.name())
        return list(set(synonyms))
    
    def augment_text(self, text):
        """
        文本增强函数
        
        输入:
            text: str, 原始文本
            
        输出:
            str: 增强后的文本
            
        功能:
            通过同义词替换进行文本增强
        """
        if not cfg.DATA.TEXT_AUGMENT:
            return text
            
        words = text.split()
        num_words = len(words)
        num_to_replace = max(1, int(num_words * cfg.DATA.TEXT_REPLACE_RATIO))
        
        # 随机选择要替换的词的位置
        replace_pos = random.sample(range(num_words), min(num_to_replace, num_words))
        
        for pos in replace_pos:
            word = words[pos]
            synonyms = self.get_synonyms(word)
            if synonyms:  # 如果有同义词
                words[pos] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            # 处理文本
            text = self.text_data[idx]
            if self.augment and cfg.DATA.TEXT_AUGMENT and not self.is_test:
                text = self.augment_text(text)
                
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 处理图像
            image_path = self.image_paths[idx]
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"警告：无法加载图像 {image_path}，使用空白图像代替")
                image = Image.new('RGB', (cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE), 'white')
            
            if self.augment and cfg.DATA.IMAGE_AUGMENT and not self.is_test:
                image = self.augment_transform(image)
            else:
                image = self.basic_transform(image)
            
            # 获取标签或guid
            if self.is_test:
                return {
                    'text': {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    },
                    'image': image,
                    'guid': self.labels[idx]  # 测试集中labels存储的是guid
                }
            else:
                label = torch.tensor(self.labels[idx])
                return {
                    'text': {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    },
                    'image': image,
                    'labels': label
                }
                
        except Exception as e:
            print(f"警告：处理样本 {idx} 时出错：{str(e)}")
            # 返回一个默认样本
            if self.is_test:
                return {
                    'text': {
                        'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                        'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
                    },
                    'image': torch.zeros(3, cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE),
                    'guid': self.labels[idx]
                }
            else:
                return {
                    'text': {
                        'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                        'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
                    },
                    'image': torch.zeros(3, cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE),
                    'labels': torch.tensor(1)  # 默认为neutral类
                }

def create_balanced_sampler(labels):
    """
    创建平衡采样器
    
    输入:
        labels: list, 标签列表
        
    输出:
        sampler: WeightedRandomSampler, 加权随机采样器
        
    功能:
        创建用于平衡数据集的采样器
    """
    if not cfg.DATA.USE_WEIGHTED_SAMPLER:
        return None
        
    # 计算类别权重
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # 计算每个样本的权重
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights = [class_weights[label] for label in labels]
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler

def create_augmented_data(text_data, image_paths, labels):
    """
    创建增强数据
    
    输入:
        text_data: list, 文本数据列表
        image_paths: list, 图像路径列表
        labels: list, 标签列表
        
    输出:
        tuple: (增强后的文本列表, 增强后的图像路径列表, 增强后的标签列表)
        
    功能:
        通过数据增强扩充数据集
    """
    if not cfg.DATA.USE_DATA_AUGMENTATION:
        return text_data, image_paths, labels
        
    print("正在进行数据平衡增强...")
    print(f"原始类别分布:")
    label_counts = np.bincount(labels)
    for label, count in enumerate(label_counts):
        print(f"类别 {label}: {count} ({count/len(labels)*100:.2f}%)")
    
    augmented_text = []
    augmented_images = []
    augmented_labels = []
    
    # 统计每个类别的样本数
    max_count = max(label_counts)
    
    # 对每个类别进行增强
    for label in range(len(label_counts)):
        # 找出该类别的所有样本索引
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # 计算需要增强的数量
        num_augment = max_count - label_counts[label]
        
        if num_augment > 0:
            print(f"类别 {label} 需要增强 {num_augment} 个样本")
            # 随机选择样本进行增强
            augment_indices = np.random.choice(indices, size=num_augment, replace=True)
            
            for idx in augment_indices:
                augmented_text.append(text_data[idx])
                augmented_images.append(image_paths[idx])
                augmented_labels.append(label)
    
    # 合并原始数据和增强数据
    all_text = text_data + augmented_text
    all_images = image_paths + augmented_images
    all_labels = labels + augmented_labels
    
    print("\n增强后的类别分布:")
    final_counts = np.bincount(all_labels)
    for label, count in enumerate(final_counts):
        print(f"类别 {label}: {count} ({count/len(all_labels)*100:.2f}%)")
    
    return all_text, all_images, all_labels

def create_data_loaders(train_text, train_images, train_labels, 
                      val_text=None, val_images=None, val_labels=None, 
                      batch_size=None, is_test=False):
    """创建数据加载器
    Args:
        train_text: 训练集/测试集文本数据
        train_images: 训练集/测试集图像路径
        train_labels: 训练集标签/测试集guid
        val_text: 验证集文本数据（可选）
        val_images: 验证集图像路径（可选）
        val_labels: 验证集标签（可选）
        batch_size: 批次大小
        is_test: 是否为测试集模式
    """
    # 使用配置文件中的参数
    batch_size = batch_size or cfg.TRAINING.BATCH_SIZE
    
    if is_test:
        # 创建测试集数据加载器
        test_dataset = MultiModalDataset(
            train_text,
            train_images,
            train_labels,
            augment=False,
            is_test=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True
        )
        
        return test_loader
    
    else:
        print(f"原始训练集大小: {len(train_labels)}")
        
        # 只对训练集进行数据平衡增强
        if cfg.DATA.USE_DATA_AUGMENTATION:
            print("使用数据平衡增强...")
            train_text, train_images, train_labels = create_augmented_data(
                train_text, train_images, train_labels
            )
            print(f"增强后的训练集大小: {len(train_labels)}")
        else:
            print("不使用数据平衡增强...")
        
        # 创建训练集
        train_dataset = MultiModalDataset(
            train_text,
            train_images,
            train_labels,
            augment=True
        )
        
        if cfg.DATA.USE_WEIGHTED_SAMPLER:
            print("使用加权采样器...")
            sampler = create_balanced_sampler(train_labels)
            shuffle = False
        else:
            print("使用随机打乱...")
            sampler = None
            shuffle = True
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True
        )
        
        # 创建验证集
        if val_text is not None and val_images is not None and val_labels is not None:
            print(f"验证集大小: {len(val_labels)}")
            val_dataset = MultiModalDataset(
                val_text,
                val_images,
                val_labels,
                augment=False
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg.DATA.NUM_WORKERS,
                pin_memory=True
            )
            
            return train_loader, val_loader
        
        return train_loader 