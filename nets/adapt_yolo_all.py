import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo import YOLOPAFPN, FeatureFusion
import torch.utils.checkpoint as checkpoint
from .resnet12_2 import resnet12


class ADAPT_YoloBody(nn.Module):
    """
    为ADAPT适配的YoloBody模型
    支持动态的分类数量设置
    """
    
    def __init__(self, imgc, imgsz, num_classes=5, phi='s', pretrained_path=None, freeze_backbone=True):
        super(ADAPT_YoloBody, self).__init__()
        
        self.imgc = imgc
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # 使用YOLOPAFPN作为backbone
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        
        depth = depth_dict[phi]
        width = width_dict[phi]
        depthwise = True if phi == 'nano' else False
        
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.fusion_module = FeatureFusion()
        # self.backbone = resnet12()
        # 动态分类器，可以根据num_classes调整
        # self.classifier = nn.Sequential(
        #     nn.Linear(640, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes)
        # )
        self.classifier = nn.Linear(224, num_classes)

        # 加载预训练权重
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)
        
        # 冻结主干网络
        if self.freeze_backbone:
            self._freeze_backbone()
        
        # 初始化分类器权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 处理可能的键名不匹配
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤掉不匹配的键
            model_dict = self.state_dict()
            filtered_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                elif k.startswith('backbone.') and k in model_dict:
                    filtered_dict[k] = v
            
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"成功加载预训练权重: {pretrained_path}")
            print(f"加载了 {len(filtered_dict)} 个权重参数")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化")
    
    def _freeze_backbone(self):
        """冻结主干网络参数"""
        frozen_params = 0
        for name, param in self.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"冻结了 {frozen_params} 个主干网络参数")
        print(f"可训练参数: {trainable_params} 个")
    
    def get_trainable_parameters(self):
        """获取可训练参数（主要用于ADAPT的梯度计算）"""
        return [param for param in self.parameters() if param.requires_grad]
    
    def forward(self, x, vars=None, bn_training=True):
        """
        前向传播
        Args:
            x: [B, C, H, W] 输入图像
            vars: 可选参数列表，用于ADAPT的参数替换
            bn_training: 是否训练BN层
        Returns:
            logits: [B, num_classes] 分类logits
        """
        if vars is not None:
            # 使用提供的参数进行前向传播（ADAPT微观层使用）
            # return self.forward_with_params(x, vars, bn_training)
            return self.forward_with_params_simple(x, vars, bn_training=True)
        
        # 正常前向传播
        # features = self.backbone(x)
        P3_outs, P4_outs, P5_outs = self.backbone(x)
        features = self.fusion_module(P3_outs, P4_outs, P5_outs)
        logits = self.classifier(features)
        
        return logits

    def forward_with_params_simple(self, x, params=None, bn_training=True):
        """使用functional_call的简化实现"""

        if params is None:
            return self.forward(x)

        # 处理生成器输入
        if hasattr(params, '__next__'):  # 如果是生成器
            params = list(params)  # 转换为列表

        # 处理参数格式
        if isinstance(params, (list, tuple)):
            named_params = dict(self.named_parameters())
            param_dict = {}
            for i, param in enumerate(params):
                if i < len(named_params):
                    name = list(named_params.keys())[i]
                    param_dict[name] = param
        else:
            param_dict = params

            # 定义前向传播函数
        def custom_forward(x_tensor):
            return torch.func.functional_call(self, param_dict, (x_tensor,))

        # 使用梯度检查点
        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        # 使用functional_call
        # return torch.func.functional_call(self, param_dict, (x,))

    def forward_with_params(self, x, vars, bn_training=True):
        """使用指定参数进行前向传播"""
        # print(f"输入x形状: {x.shape}")
        # 处理不同类型的vars输入
        if vars is not None:
            if hasattr(vars, '__next__'):  # 生成器类型
                vars = list(vars)
            elif isinstance(vars, (tuple, set)):  # 元组或集合
                vars = list(vars)

        # 这里简化处理，实际应该根据参数结构进行替换
        # 由于主干网络被冻结，我们只关心分类器的参数
        # with torch.no_grad():
        P3_outs, P4_outs, P5_outs = self.backbone(x)
        # print(f"特征图形状: {features.shape}")
        # 使用分类器的参数
        if len(vars) >= 8:  # 分类器有4个参数（2个线性层的weight和bias）
            # 手动应用分类器参数
            new_vars = vars[-8:]
            fusion_vars = new_vars[:6]  # fusion_module的4个参数
            classifier_vars = new_vars[6:8]  # 分类器的4个参数（单层）

            # 手动应用fusion_module参数
            weight_p5, bias_p5, weight_p4, bias_p4, weight_p3, bias_p3= fusion_vars
            # Global Average Pooling
            p3_gap = F.adaptive_avg_pool2d(P3_outs, (1, 1)).squeeze(-1).squeeze(-1)
            p4_gap = F.adaptive_avg_pool2d(P4_outs, (1, 1)).squeeze(-1).squeeze(-1)
            p5_gap = F.adaptive_avg_pool2d(P5_outs, (1, 1)).squeeze(-1).squeeze(-1)
            # 应用fusion_module的线性变换
            p5_embed = F.relu(F.linear(p5_gap, weight_p5, bias_p5))
            p4_embed = F.relu(F.linear(p4_gap, weight_p4, bias_p4))
            p3_embed = F.relu(F.linear(p3_gap, weight_p3, bias_p3))  # p3的映射

            # 拼接特征
            features = torch.cat([p5_embed, p4_embed, p3_embed], dim=1)

            # 手动应用单层分类器参数
            weight_cls, bias_cls = classifier_vars[0], classifier_vars[1]
            logits = F.linear(features, weight_cls, bias_cls)

            return logits

        # 默认使用当前参数
        features = self.fusion_module(P3_outs, P4_outs, P5_outs)
        return self.classifier(features)


    # def forward_with_params(self, x, vars, bn_training=True):
    #     """使用指定参数进行前向传播"""
    #     # print(f"输入x形状: {x.shape}")
    #     # 处理不同类型的vars输入
    #     if vars is not None:
    #         if hasattr(vars, '__next__'):  # 生成器类型
    #             vars = list(vars)
    #         elif isinstance(vars, (tuple, set)):  # 元组或集合
    #             vars = list(vars)
    #
    #     # 这里简化处理，实际应该根据参数结构进行替换
    #     # 由于主干网络被冻结，我们只关心分类器的参数
    #
    #     P3_outs, P4_outs, P5_outs = self.backbone(x)
    #     features = self.fusion_module(P3_outs, P4_outs, P5_outs)
    #     # print(f"特征图形状: {features.shape}")
    #     # 使用分类器的参数
    #     if len(vars) >= 8:  # 分类器有4个参数（2个线性层的weight和bias）
    #         # 手动应用分类器参数
    #
    #         # 取最后4个参数作为分类器参数
    #         classifier_vars = vars[-4:]
    #         weight1, bias1, weight2, bias2 = classifier_vars
    #
    #         x = features
    #         # print(f"len(vars)大小：{len(vars)}")
    #         # 第一层
    #         # weight1, bias1 = vars[0], vars[1]
    #         x = F.linear(x, weight1, bias1)
    #         x = F.relu(x, inplace=True)
    #         x = F.dropout(x, p=0.5, training=bn_training)
    #
    #         # 第二层
    #         # weight2, bias2 = vars[2], vars[3]
    #         logits = F.linear(x, weight2, bias2)
    #
    #         return logits
    #
    #     # 默认使用当前参数
    #     return self.classifier(features)
    #
    def get_parameters(self):
        """获取所有可训练参数，用于ADAPT的梯度计算"""
        return self.get_trainable_parameters()
    
    def adapt_classifier(self, new_num_classes):
        """
        动态调整分类器以适应不同的way数量
        Args:
            new_num_classes: 新的分类数量
        """
        old_classifier = self.classifier
        
        # 创建新的分类器
        new_classifier = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, new_num_classes)
        )
        
        # 复制共享层的权重
        if old_classifier[0].weight.data.shape == new_classifier[0].weight.data.shape:
            new_classifier[0].weight.data = old_classifier[0].weight.data.clone()
            new_classifier[0].bias.data = old_classifier[0].bias.data.clone()
        
        self.classifier = new_classifier
        self.num_classes = new_num_classes
        
        return new_classifier


class ADAPT_YoloWrapper(nn.Module):
    """
    ADAPT兼容的YoloBody包装器
    模拟learner.py的接口风格
    """
    
    def __init__(self, imgc, imgsz, num_classes=5, phi='s', pretrained_path=None, freeze_backbone=True):
        super(ADAPT_YoloWrapper, self).__init__()
        
        self.model = ADAPT_YoloBody(imgc, imgsz, num_classes, phi, pretrained_path, freeze_backbone)
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        
        # 只收集可训练参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bn' in name and 'running' not in name:
                    self.vars_bn.append(param)
                elif 'running_mean' in name or 'running_var' in name:
                    self.vars_bn.append(param)
                else:
                    self.vars.append(param)
    
    def forward(self, x, vars=None, bn_training=True):
        """
        前向传播，支持参数替换（ADAPT的关键特性）
        """
        if vars is None:
            return self.model(x)
        
        # 保存原始参数
        original_params = []
        for param in self.model.parameters():
            original_params.append(param.data.clone())
        
        try:
            # 应用新的参数
            param_idx = 0
            for param in self.model.parameters():
                if param_idx < len(vars):
                    param.data = vars[param_idx]
                    param_idx += 1
            
            # 前向传播
            output = self.model(x)
            
        finally:
            # 恢复原始参数
            param_idx = 0
            for param in self.model.parameters():
                if param_idx < len(original_params):
                    param.data = original_params[param_idx]
                    param_idx += 1
        
        return output
    
    def parameters(self):
        """返回所有可训练参数"""
        return self.model.parameters()
    
    def get_config(self):
        """获取网络配置信息"""
        return [
            ('ADAPT_yolo', [self.model.imgc, self.model.imgsz, self.model.num_classes])
        ]


# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = ADAPT_YoloBody(imgc=3, imgsz=84, num_classes=5)
    dummy_input = torch.randn(2, 3, 84, 84)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")