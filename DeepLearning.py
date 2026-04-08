import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# 1. 数据集解析与加载 (Dataset)
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        data_root: 包含 benign, malignant, normal 文件夹的根目录
        """
        self.transform = transform
        self.samples = []
        
        # 定义类别映射
        self.class_to_idx = {'normal': 0, 'benign': 1, 'malignant': 2}
        
        print("🔍 正在扫描并解析 BUSI 数据集...")
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(data_root, cls_name)
            if not os.path.exists(cls_dir):
                continue
                
            # 找到所有原图 (排除名字里带有 _mask 的文件)
            all_files = glob.glob(os.path.join(cls_dir, '*.png'))
            img_paths = [f for f in all_files if '_mask' not in f]
            
            for img_path in img_paths:
                # 寻找对应的所有 mask 文件 (处理单个或多个病灶的情况)
                base_name = img_path.replace('.png', '')
                mask_paths = glob.glob(f"{base_name}_mask*.png")
                
                self.samples.append({
                    'image': img_path,
                    'masks': mask_paths,
                    'label': cls_idx
                })
        print(f"共找到 {len(self.samples)} 个有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 读原图 (强制单通道灰度)
        image = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE)
        
        # 2. 读取并合并 Masks
        # 如果是 normal 类别，可能没有 mask，或者 mask 是全黑的
        h, w = image.shape
        combined_mask = np.zeros((h, w), dtype=np.float32)
        
        for mask_path in sample['masks']:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                # 将 255 变成 1.0
                m = (m > 127).astype(np.float32) 
                combined_mask = np.maximum(combined_mask, m) # 叠加多病灶
                
        # 3. 数据增强 (Albumentations 同步处理原图和 Mask)
        if self.transform:
            augmented = self.transform(image=image, mask=combined_mask)
            image = augmented['image']
            combined_mask = augmented['mask']
            
        # 4. 转换标签类型
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # Albumentations 对单通道图像返回 (H,W)，我们需要手动增加通道维度变为 (1,H,W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(combined_mask.shape) == 2:
            combined_mask = combined_mask.unsqueeze(0)
            
        return image.float(), label, combined_mask.float()

# ==========================================
# 2. 多任务网络架构 (ResNet34 + U-Net Decoder)
# ==========================================
class UltrasoundMultiTaskNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UltrasoundMultiTaskNet, self).__init__()
        
        # --- 编码器 (Encoder) ---
        resnet = models.resnet34(pretrained=True)
        # 修改第一层，接受 1 通道灰度图
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu
        )
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1) # 64
        self.encoder3 = resnet.layer2 # 128
        self.encoder4 = resnet.layer3 # 256
        self.encoder5 = resnet.layer4 # 512

        # --- 分类头 (Classification Head) ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # --- 分割头 (Segmentation Head) ---
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._decoder_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._decoder_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._decoder_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = self._decoder_block(128, 64)

        self.final_upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # 注意：这里去掉了 Sigmoid，交由 BCEWithLogitsLoss 处理，更稳定
        self.seg_output = nn.Conv2d(64, 1, kernel_size=1) 

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # 分类预测
        cls_feat = self.avgpool(e5)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_out = self.fc(cls_feat)

        # 分割预测 (带跳跃连接)
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        final = self.final_upconv(d1)
        seg_out = self.seg_output(final) # Raw Logits

        return cls_out, seg_out

# ==========================================
# 3. 主程序：训练流水线
# ==========================================
def main():
    # 超参数设置
    DATA_DIR = "./Dataset_BUSI_with_GT" # 
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {DEVICE}")

    # 数据增强策略 (使用 Albumentations)
    train_transform = A.Compose([
        A.Resize(256, 256),              # 统一尺寸为 256x256
        A.HorizontalFlip(p=0.5),         # 50%概率水平翻转
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), # 轻微平移、缩放、旋转
        # 注意：不再做图像去噪或模糊增强，保留超声原味
        A.Normalize(mean=(0.5,), std=(0.5,)), # 归一化到 [-1, 1]
        ToTensorV2()
    ])

    # 加载数据
    dataset = BUSIDataset(DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型、损失函数和优化器
    model = UltrasoundMultiTaskNet(num_classes=3).to(DEVICE)
    
    criterion_cls = nn.CrossEntropyLoss()
    # 使用 BCEWithLogitsLoss，它内部集成了 Sigmoid，数值计算更稳定
    criterion_seg = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # 训练循环
    print("开始训练多任务网络...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        correct_cls = 0
        total_cls = 0
        
      
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for images, labels, masks in pbar:
            images, labels, masks = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            cls_out, seg_out = model(images)

            loss_cls = criterion_cls(cls_out, labels)
            loss_seg = criterion_seg(seg_out, masks)
            # 联合损失 
            loss = 0.5 * loss_cls + 0.5 * loss_seg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += loss_cls.item()
            running_seg_loss += loss_seg.item()
            
            # 计算分类准确率
            _, predicted = torch.max(cls_out.data, 1)
            total_cls += labels.size(0)
            correct_cls += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Total Loss': f"{loss.item():.4f}", 
                'Cls Acc': f"{100 * correct_cls / total_cls:.2f}%"
            })

        # 打印 Epoch 总结
        epoch_loss = running_loss / len(dataloader)
        print(f" Epoch {epoch+1} 总结 -> Avg Loss: {epoch_loss:.4f} | Avg Cls Loss: {running_cls_loss/len(dataloader):.4f} | Avg Seg Loss: {running_seg_loss/len(dataloader):.4f}")

    # 保存最终训练好的模型 
    torch.save(model.state_dict(), "ultrasound_multitask_busi.pth")
    print(" 训练完成！模型已保存为 ultrasound_multitask_busi.pth")

if __name__ == '__main__':
    main()
