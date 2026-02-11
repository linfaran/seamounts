import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from thop import profile
import wandb
import matplotlib.pyplot as plt
import numpy as np

# === 新增：导入五个模型（把这五个单文件放到同目录或可导入路径中） ===
from unet import UNet
from unet import ScConv_UNet
from unet import MultiResUnet
from unet import UnetPlusPlus
from unet import DenseU_Net
from unet import build_doubleunet
from unet import AttU_Net
from unet import R2AttU_Net

from unet import SegmenterMini
from unet import PSPNet
from unet import DeepLabV3Plus
from unet import Mask2FormerMini
from unet import TransUNet

from evaluate import evaluate
from unet import UNet, ScConv_UNet, MultiResUnet, UnetPlusPlus, DenseU_Net, build_doubleunet
from unet import AttU_Net, R2U_Net, R2AttU_Net
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

class BinaryToTwoAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.proj = nn.Conv2d(1, 2, kernel_size=1, bias=True)
    def forward(self, x):
        y = self.base(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        return self.proj(y)

# 定义所有数据集名称和对应文件夹
DATASETS = ['topo', 'dem4', 'geo', 'globe', 'ibcso']

# 根目录
DATA_ROOT = Path('./data')
CHECKPOINT_ROOT = Path('./checkpoints')
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

# === 新增：要跑的 5 个模型列表（名字 -> 构造函数） ===
MODEL_BUILDERS = {
    # # ===== 你之前的模型 =====
    'UNet':           lambda n_in, n_cls, args=None: UNet(n_channels=n_in, n_classes=n_cls, bilinear=(args.bilinear if args else False)),
    'ScConv_UNet':    lambda n_in, n_cls, args=None: ScConv_UNet(n_channels=n_in, n_classes=n_cls, bilinear=(args.bilinear if args else False)),
    'MultiResUnet':   lambda n_in, n_cls, args=None: MultiResUnet(n_channels=n_in, n_classes=n_cls, alpha=1.67),
    # UnetPlusPlus 的 __init__ 是 (num_classes, deep_supervision=False)，且 n_channels=3 写死
    'UnetPlusPlus':   lambda n_in, n_cls, args=None: (
        (print("[Warn] UnetPlusPlus: n_channels fixed to 3 in your class; got --in-channels =", n_in) or True)
        and UnetPlusPlus(num_classes=n_cls, deep_supervision=False)
    ),
    # 下面两个构造函数可能与你本地略有不同；如果签名不匹配会在 try/except 里捕获并跳过
    'DenseU_Net':     lambda n_in, n_cls, args=None: DenseU_Net(n_in, n_cls) if 'DenseU_Net' in globals() else None,
    'DoubleUNet':     lambda n_in, n_cls, args=None: BinaryToTwoAdapter(build_doubleunet()),
    'AttU_Net':       lambda n_in, n_cls, args=None: AttU_Net(n_channels=n_in, output_ch=n_cls),
    'R2AttU_Net':     lambda n_in, n_cls, args=None: R2AttU_Net(n_channels=n_in, output_ch=n_cls, t=2),

    # ===== 我给你的 5 个新模型 =====
    'DeepLabv3Plus':  lambda n_in, n_cls, args=None: DeepLabV3Plus(n_channels=n_in, n_classes=n_cls, aspp_out=256, low_ch=48),
    'PSPNet':         lambda n_in, n_cls, args=None: PSPNet(n_channels=n_in, n_classes=n_cls, feat=128, ppm_out=128, bins=(1,2,3,6)),
    'TransUNet':      lambda n_in, n_cls, args=None: TransUNet(n_channels=n_in, n_classes=n_cls, embed_dim=256, patch=16, vit_depth=4),
    'SegmenterMini':  lambda n_in, n_cls, args=None: SegmenterMini(n_channels=n_in, n_classes=n_cls, embed_dim=256, patch=16, depth=6),
    'Mask2FormerMini':lambda n_in, n_cls, args=None: Mask2FormerMini(n_channels=n_in, n_classes=n_cls, dim=256, num_queries=100, dec_layers=3),
}

def attach_meta(model, n_channels, n_classes):
    """确保模型有 n_channels / n_classes 属性，兼容现有训练代码"""
    if not hasattr(model, 'n_channels'):
        model.n_channels = n_channels
    if not hasattr(model, 'n_classes'):
        model.n_classes = n_classes
    return model

def train_model(
        model,
        device,
        dataset_name: str,
        model_name: str,                 # === 新增：记录模型名 ===
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-7,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 设置数据路径
    dir_img = DATA_ROOT / f'imgs_{dataset_name}'
    dir_mask = DATA_ROOT / f'masks_{dataset_name}'

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    # 3. DataLoaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # WandB
    experiment = wandb.init(
        project='U-Net',
        name=f'{dataset_name}-{model_name}',   # === 改：名称包含模型名 ===
        resume='allow',
        reinit=True,
        anonymous='must',
        mode='offline'
    )
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            img_scale=img_scale,
            amp=amp,
            dataset=dataset_name,
            model=model_name
        )
    )

    # Optimizer etc.
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # 以 val_loss 调整
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # FLOPs
    dummy_input = torch.randn(1, model.n_channels, 512, 512, device=device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"### {dataset_name}-{model_name} Model complexity ###\n"
          f"FLOPs: {flops / 1e9:.3f} GFLOPs\n"
          f"Params: {params / 1e6:.3f} M")

    # 存储 metrics
    train_losses, val_losses = [], []

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'{dataset_name}-{model_name} Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type!='mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # ① 空间尺寸对齐（非常关键，兼容 DoubleUNet 等模型）
                    if masks_pred.shape[-2:] != true_masks.shape[-2:]:
                        masks_pred = F.interpolate(
                            masks_pred, size=true_masks.shape[-2:], mode='bilinear', align_corners=False
                        )

                    # ② （可选）通道数防御：若模型最后一层通道不等于 n_classes，尽早报错更清晰
                    if masks_pred.shape[1] != model.n_classes:
                        raise RuntimeError(
                            f"Model logits channels ({masks_pred.shape[1]}) != n_classes ({model.n_classes}). "
                            "请检查 DoubleUNet 的最后一层输出通道或用我之前给的 _set_num_classes_if_needed() 改头。"
                        )

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            torch.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks.long(), model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # 记录 metrics
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                with torch.autocast(device.type if device.type!='mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # ① 空间尺寸对齐（非常关键，兼容 DoubleUNet 等模型）
                    if masks_pred.shape[-2:] != true_masks.shape[-2:]:
                        masks_pred = F.interpolate(
                            masks_pred, size=true_masks.shape[-2:], mode='bilinear', align_corners=False
                        )

                    # ② （可选）通道数防御：若模型最后一层通道不等于 n_classes，尽早报错更清晰
                    if masks_pred.shape[1] != model.n_classes:
                        raise RuntimeError(
                            f"Model logits channels ({masks_pred.shape[1]}) != n_classes ({model.n_classes}). "
                            "请检查 DoubleUNet 的最后一层输出通道或用我之前给的 _set_num_classes_if_needed() 改头。"
                        )

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            torch.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks.long(), model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        # scheduler and log
        scheduler.step(avg_val_loss)
        experiment.log({
            'epoch': epoch,
            'validation loss': avg_val_loss,
            'learning rate': optimizer.param_groups[0]['lr']
        })

        # 只在最后 epoch 保存模型
        if save_checkpoint and epoch == epochs:
            save_path = CHECKPOINT_ROOT / f'{dataset_name}_{model_name}_last_epoch.pth'
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(save_path))
            logging.info(f'{dataset_name}-{model_name} checkpoint saved: {save_path}')

    # 保存 metrics 和曲线
    metrics = {'train_losses': train_losses, 'val_losses': val_losses}
    torch.save(metrics, str(CHECKPOINT_ROOT / f'{dataset_name}_{model_name}_metrics.pth'))

    epochs_range = range(1, epochs + 1)
    import matplotlib
    matplotlib.use('Agg')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{dataset_name}-{model_name} Loss')
    plt.legend()
    fig_path = CHECKPOINT_ROOT / f'{dataset_name}_{model_name}_loss_curve.png'
    plt.savefig(str(fig_path))
    plt.close()
    print(f"[Saved] Loss curve → {fig_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Train on multiple datasets & models')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Image downscale')
    parser.add_argument('--validation', '-v', type=float, default=10.0, help='Validation percent')
    parser.add_argument('--amp', action='store_true', help='Mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling in UNet-like decoders')  # <—— 新增
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 对每个数据集、每个模型都训练一遍
    for ds in DATASETS:
        for model_name, builder in MODEL_BUILDERS.items():
            print(f"\n=== Dataset: {ds} | Model: {model_name} ===")
            try:
                model = builder(args.in_channels, args.classes, args)  # <—— 把 args 传进去（部分 builder 会用到）
                if model is None:
                    print(f"[Skip] {model_name} not available (builder returned None).")
                    continue
            except TypeError:
                # 向后兼容：如果某个 builder 只接受两个参数
                model = builder(args.in_channels, args.classes)

            model = attach_meta(model, n_channels=args.in_channels, n_classes=args.classes)
            model = model.to(device=device, memory_format=torch.channels_last)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"{ds}-{model_name} Total parameters: {total_params}")

            train_model(
                model=model,
                device=device,
                dataset_name=ds,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                img_scale=args.scale,
                val_percent=args.validation / 100.0,
                amp=args.amp
            )
