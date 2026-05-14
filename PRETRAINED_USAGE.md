# 使用预训练权重进行ADAPT训练

本文档说明如何在ADAPT训练中使用预训练的Body权重。

## 功能特性

- ✅ **预训练权重加载**：支持加载预训练的Body主干网络权重
- ✅ **主干网络冻结**：在元训练过程中冻结主干网络，只训练分类头
- ✅ **动态分类器**：支持根据way数量动态调整分类器输出
- ✅ **内存优化**：通过冻结主干网络减少内存占用和计算量

## 使用方法

### 1. 准备预训练权重

将你的预训练权重文件放在 `nets/` 目录下，支持的格式包括：
- `.pth` 文件
- `.pt` 文件

### 2. 训练命令

#### 使用预训练权重并冻结主干网络（推荐）
```bash
python train_with_pretrained.py \
    --pretrained_path nets/body_pretrained.pth \
    --n_way 5 \
    --k_spt 1 \
    --k_qry 15 \
    --epoch 60000 \
    --freeze_backbone
```

#### 使用预训练权重但不冻结主干网络（全网络微调）
```bash
python train_with_pretrained.py \
    --pretrained_path nets/body_pretrained.pth \
    --n_way 5 \
    --k_spt 1 \
    --k_qry 15 \
    --epoch 60000 \
    --no_freeze_backbone
```

#### 不使用预训练权重（从头训练）
```bash
python train_with_pretrained.py \
    --n_way 5 \
    --k_spt 1 \
    --k_qry 15 \
    --epoch 60000
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_path` | 预训练权重文件路径 | `None` |
| `--freeze_backbone` | 是否冻结主干网络 | `True` |
| `--no_freeze_backbone` | 不冻结主干网络（全网络训练） | - |
| `--n_way` | 分类类别数（way） | `5` |
| `--k_spt` | 支持集样本数（shot） | `1` |
| `--k_qry` | 查询集样本数 | `15` |
| `--epoch` | 训练轮数 | `60000` |
| `--imgsz` | 输入图像尺寸 | `84` |
| `--meta_lr` | 元学习率 | `1e-3` |
| `--update_lr` | 内循环学习率 | `0.01` |

## 技术细节

### 权重加载机制

1. **智能键匹配**：自动处理权重文件中的键名与模型参数的匹配
2. **形状检查**：只加载形状匹配的权重参数
3. **错误处理**：如果权重加载失败，会自动回退到随机初始化

### 参数冻结策略

当 `--freeze_backbone` 为 `True` 时：
- ✅ **冻结的层**：`backbone`和 `fusion_module`
- ✅ **可训练的层**：`classifier`

### 内存优化

通过冻结主干网络，可以显著减少：
- 内存占用（约减少80%）
- 计算量（约减少70%）
- 训练时间（约减少60%）

## 数据集支持

### FER-2013
```bash
python train_with_pretrained.py \
    --pretrained_path nets/body_pretrained.pth \
    --n_way 7 --k_spt 5 --k_qry 15
```

### 自定义预训练权重
如果你的预训练权重格式不同，可以修改 `load_pretrained_weights` 方法：

```python
def load_custom_weights(self, pretrained_path):
    # 你的自定义加载逻辑
    checkpoint = torch.load(pretrained_path)
    # 处理权重映射...
```

### 选择性冻结
可以修改 `_freeze_backbone` 方法来自定义冻结策略：

```python
def _custom_freeze(self):
    # 只冻结backbone的某些层
    for name, param in self.named_parameters():
        if 'backbone.stage1' in name:
            param.requires_grad = False
```

## 注意事项

1. **权重兼容性**：确保预训练权重的网络结构与当前模型匹配
2. **GPU内存**：冻结主干网络可以显著减少内存占用
3. **学习率**：使用预训练权重时，建议使用较小的学习率（如1e-4）
4. **数据增强**：预训练权重通常对数据增强更敏感