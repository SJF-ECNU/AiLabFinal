import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
import torch.nn.functional as F
from config import Config as cfg

class CrossAttention(nn.Module):
    """
    交叉注意力模块
    
    输入:
        dim: int, 特征维度
        
    输出:
        None
    
    功能:
        实现两个模态特征之间的交叉注意力机制
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(cfg.MODEL.ATTENTION_DROPOUT)
        
    def forward(self, x1, x2):
        """
        前向传播函数
        
        输入:
            x1: tensor, 第一个模态的特征
            x2: tensor, 第二个模态的特征
            
        输出:
            out: tensor, 注意力加权后的特征
            
        功能:
            计算两个模态特征间的注意力权重并融合
        """
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        
        out = attn @ v
        return out

class FeatureExtractor(nn.Module):
    """
    特征提取器模块
    
    输入:
        None
        
    输出:
        None
        
    功能:
        提取和融合文本与图像特征
    """
    def __init__(self):
        super().__init__()
        
        # 文本特征提取器 (BERT)
        self.text_encoder = BertModel.from_pretrained(cfg.MODEL.BERT_NAME)
        
        # 图像特征提取器 (ViT)
        self.image_encoder = ViTModel.from_pretrained(cfg.MODEL.VIT_NAME)
        
        # 特征维度
        self.text_dim = cfg.MODEL.TEXT_DIM
        self.image_dim = cfg.MODEL.IMAGE_DIM
        
        # 初始化特征属性
        self.text_features = None
        self.image_features = None
        self.text_attended = None
        self.image_attended = None
        self.fusion_weights = None
        
        # 特征转换层
        self.text_transform = nn.Sequential(
            nn.Linear(self.text_dim, cfg.MODEL.HIDDEN_DIM),
            nn.LayerNorm(cfg.MODEL.HIDDEN_DIM) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATES[0])
        )
        
        self.image_transform = nn.Sequential(
            nn.Linear(self.image_dim, cfg.MODEL.HIDDEN_DIM),
            nn.LayerNorm(cfg.MODEL.HIDDEN_DIM) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATES[0])
        )
        
        # 交叉注意力层
        self.text_to_image_attention = CrossAttention(cfg.MODEL.HIDDEN_DIM)
        self.image_to_text_attention = CrossAttention(cfg.MODEL.HIDDEN_DIM)
        
        # 动态特征融合
        fusion_input_dim = cfg.MODEL.HIDDEN_DIM * 4
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.MODEL.HIDDEN_DIM),
            nn.LayerNorm(cfg.MODEL.HIDDEN_DIM) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATES[0]),
            nn.Linear(cfg.MODEL.HIDDEN_DIM, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, text_inputs=None, image_inputs=None):
        """
        前向传播函数
        
        输入:
            text_inputs: dict, 文本输入
            image_inputs: tensor, 图像输入
            
        输出:
            fused_features: tensor, 融合后的特征
            
        功能:
            提取并融合文本和图像特征
        """
        self.text_features = None
        self.image_features = None
        
        # 文本特征提取
        if text_inputs is not None:
            text_outputs = self.text_encoder(**text_inputs)
            self.text_features = text_outputs.pooler_output
            self.text_features = self.text_transform(self.text_features)
            
        # 图像特征提取
        if image_inputs is not None:
            image_outputs = self.image_encoder(image_inputs)
            self.image_features = image_outputs.pooler_output
            self.image_features = self.image_transform(self.image_features)
            
        # 特征融合
        if self.text_features is not None and self.image_features is not None:
            # 使用交叉注意力
            self.text_attended = self.text_to_image_attention(self.text_features.unsqueeze(1), self.image_features.unsqueeze(1)).squeeze(1)
            self.image_attended = self.image_to_text_attention(self.image_features.unsqueeze(1), self.text_features.unsqueeze(1)).squeeze(1)
            
            # 计算动态融合权重
            all_features = torch.cat([
                self.text_features,
                self.image_features,
                self.text_attended,
                self.image_attended
            ], dim=1)
            
            self.fusion_weights = self.fusion_gate(all_features)
            
            # 加权融合所有特征
            fused_features = torch.cat([
                self.text_features * self.fusion_weights[:, 0:1],
                self.image_features * self.fusion_weights[:, 1:2],
                self.text_attended * self.fusion_weights[:, 2:3],
                self.image_attended * self.fusion_weights[:, 3:4]
            ], dim=1)
            
        elif self.text_features is not None:
            fused_features = torch.cat([self.text_features, torch.zeros_like(self.text_features)]*2, dim=1)
        else:
            fused_features = torch.cat([torch.zeros_like(self.image_features), self.image_features]*2, dim=1)
            
        return fused_features

class MultiModalClassifier(nn.Module):
    """
    多模态分类器
    
    输入:
        None
        
    输出:
        None
        
    功能:
        结合文本和图像特征进行情感分类
    """
    def __init__(self):
        super().__init__()
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(cfg.MODEL.FUSION_DIM, cfg.MODEL.FUSION_DIM),
            nn.LayerNorm(cfg.MODEL.FUSION_DIM) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FUSION_DIM, 1)
        )
        
        # Neutral检测分支
        neutral_layers = []
        prev_dim = cfg.MODEL.FUSION_DIM
        for i, dim in enumerate(cfg.MODEL.CLASSIFIER_DIMS[:-1]):
            neutral_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(cfg.MODEL.DROPOUT_RATES[min(i, len(cfg.MODEL.DROPOUT_RATES)-1)])
            ])
            prev_dim = dim
        neutral_layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        self.neutral_detector = nn.Sequential(*neutral_layers)
        
        # 情感分类分支
        emotion_layers = []
        prev_dim = cfg.MODEL.FUSION_DIM
        for i, dim in enumerate(cfg.MODEL.CLASSIFIER_DIMS[:-1]):
            emotion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim) if cfg.MODEL.USE_LAYER_NORM else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(cfg.MODEL.DROPOUT_RATES[min(i, len(cfg.MODEL.DROPOUT_RATES)-1)])
            ])
            prev_dim = dim
        emotion_layers.append(nn.Linear(prev_dim, 2))
        self.emotion_classifier = nn.Sequential(*emotion_layers)
    
    def forward(self, text_inputs=None, image_inputs=None):
        """
        前向传播函数
        
        输入:
            text_inputs: dict, 文本输入
            image_inputs: tensor, 图像输入
            
        输出:
            output: tensor, 分类预测结果
            
        功能:
            对输入进行多模态情感分类
        """
        # 提取特征
        features = self.feature_extractor(text_inputs, image_inputs)
        
        # 计算注意力权重
        attention_weights = torch.sigmoid(self.attention(features))
        attended_features = features * attention_weights
        
        # Neutral检测
        neutral_prob = self.neutral_detector(attended_features)
        
        # 情感分类
        emotion_logits = self.emotion_classifier(attended_features)
        
        # 将结果组合成三分类输出
        batch_size = features.size(0)
        output = torch.zeros(batch_size, cfg.MODEL.NUM_CLASSES).to(features.device)
        
        # Neutral概率直接作为neutral类的logit
        output[:, 1] = torch.log(neutral_prob.squeeze() + 1e-6)
        
        # 非neutral的概率分配给negative和positive
        non_neutral_prob = 1 - neutral_prob
        emotion_probs = F.softmax(emotion_logits, dim=1)
        output[:, 0] = torch.log(emotion_probs[:, 0] * non_neutral_prob.squeeze() + 1e-6)  # negative
        output[:, 2] = torch.log(emotion_probs[:, 1] * non_neutral_prob.squeeze() + 1e-6)  # positive
        
        return output

class TextOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(TextOnlyModel, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, text_inputs):
        outputs = self.text_encoder(**text_inputs)
        text_features = outputs.pooler_output
        return self.classifier(text_features)

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageOnlyModel, self).__init__()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image_inputs):
        outputs = self.image_encoder(image_inputs)
        image_features = outputs.pooler_output
        return self.classifier(image_features) 