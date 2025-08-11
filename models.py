import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # 归一化到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])


class VGGEmbedding(nn.Module):
    def __init__(self, pretrained=True, emd_size=128, feature_layer='conv5_3', pool_type='avg'):
        """
        使用VGG16模型生成128维batch级embedding

        参数:
        pretrained (bool): 是否使用预训练权重
        feature_layer (str): 选择用于embedding的特征层 ('conv5_3', 'conv4_3', 'conv3_3')
        pool_type (str): 池化类型 ('avg' 或 'max')
        """
        super(VGGEmbedding, self).__init__()
        self.emd_size = emd_size
        
        # 加载预训练的VGG16模型
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 选择特征提取层
        self.feature_layer = feature_layer
        self.feature_idx = {
            'conv3_3': 16,  # ReLU after conv3_3
            'conv4_3': 23,  # ReLU after conv4_3
            'conv5_3': 30  # ReLU after conv5_3
        }
        
        # 验证选择的层是否有效
        if feature_layer not in self.feature_idx:
            raise ValueError(f"Invalid feature layer: {feature_layer}. Choose from {list(self.feature_idx.keys())}")
        
        # 创建特征提取器（截取到指定层）
        self.feature_extractor = nn.Sequential(
            *list(self.vgg.features.children())[:self.feature_idx[feature_layer] + 1]
        )
        
        # 选择池化类型
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 根据选择的层获取输出通道数
        self.out_channels = {
            'conv3_3': 256,
            'conv4_3': 512,
            'conv5_3': 512
        }[feature_layer]
        
        # 降维到128维的1x1卷积
        self.dim_reduction = nn.Conv2d(self.out_channels, self.emd_size, kernel_size=1)
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    
    def preprocess(self, x):
        """
        替代Lambda层的预处理函数
        """
        # 1. 转换输入格式: (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 2. 归一化到[0,1]
        x = x / 255.0
        
        # 3. ImageNet标准化
        x = (x - self.mean) / self.std
        
        return x
    
    def forward(self, x):
        """
        前向传播

        输入形状: (batch, H, W, C)
        输出形状: (1, 128) 固定大小的batch级embedding
        """
        # # 转换输入格式: (batch, H, W, C) -> (batch, C, H, W)
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.preprocess(x)
        
        # 特征提取（截取到指定层）
        features = self.feature_extractor(x)
        
        # 空间池化 (固定到1x1)
        pooled = self.pool(features)
        
        # 降维到128维
        reduced = self.dim_reduction(pooled)
        
        # 展平特征: (batch, 128, 1, 1) -> (batch, 128)
        flattened = reduced.view(reduced.size(0), -1)
        
        # 转置以便在特征维度上进行池化
        transposed = flattened.transpose(0, 1)  # 形状: (128, batch)
        
        # 对整个batch进行全局池化
        batch_embedding = self.global_pool(transposed)  # 形状: (128, 1)
        
        # 转回标准形状并添加batch维度
        batch_embedding = batch_embedding.transpose(0, 1)  # 形状: (1, 128)
        
        return batch_embedding


# 使用示例
if __name__ == "__main__":
    # 创建模型 - 使用conv5_3层特征，平均池化
    model = VGGEmbedding(
        pretrained=True,
        emd_size=128,
        feature_layer='conv5_3',
        pool_type='avg'
    )
    model.eval()  # 设置为评估模式
    
    # 模拟输入: batch_size=8, 图像尺寸224x224, 3通道
    input_batch = torch.randn(8, 224, 224, 3)
    
    # 前向传播
    with torch.no_grad():
        embedding = model(input_batch)
        print("输入batch形状:", input_batch.shape)
        print("输出embedding形状:", embedding.shape)  # 应为 torch.Size([1, 128])
        print("输出示例:", embedding[0, :5])