#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py (兼容 effecient.py 的 profile 调用)
- 支持推理：--model --input --output/自动_OUT.png
- 支持不保存：--no-save
- 支持保存开关：--save-masks（等价于强制保存）
- 支持 profile：--profile-warmup --profile-iters，并打印一行 [PROFILE]
- 自动从 ckpt 文件名解析模型结构：{ds}_{Arch}_last_epoch.pth
"""

import os
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# ===== 解决 Windows/Anaconda 下 OpenMP 重复初始化导致崩溃（OMP Error #15）=====
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask

# ====== 导入所有模型类（与你训练时一致）======
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


# ---------- DoubleUNet 单通道 → 2 通道 适配器 ----------
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


# ---------- 模型构造器表（与你训练时一致） ----------
def make_builders():
    return {
        "UNet": lambda n_in, n_cls, args=None: UNet(n_channels=n_in, n_classes=n_cls, bilinear=(args.bilinear if args else False)),
        "ScConv_UNet": lambda n_in, n_cls, args=None: ScConv_UNet(n_channels=n_in, n_classes=n_cls, bilinear=(args.bilinear if args else False)),
        "MultiResUnet": lambda n_in, n_cls, args=None: MultiResUnet(n_channels=n_in, n_classes=n_cls, alpha=1.67),
        "UnetPlusPlus": lambda n_in, n_cls, args=None: UnetPlusPlus(num_classes=n_cls, deep_supervision=False),
        "DenseU_Net": lambda n_in, n_cls, args=None: DenseU_Net(n_in, n_cls) if "DenseU_Net" in globals() else None,
        "DoubleUNet": lambda n_in, n_cls, args=None: BinaryToTwoAdapter(build_doubleunet()),
        "AttU_Net": lambda n_in, n_cls, args=None: AttU_Net(n_channels=n_in, output_ch=n_cls),
        "R2AttU_Net": lambda n_in, n_cls, args=None: R2AttU_Net(n_channels=n_in, output_ch=n_cls, t=2),

        "DeepLabv3Plus": lambda n_in, n_cls, args=None: DeepLabV3Plus(n_channels=n_in, n_classes=n_cls, aspp_out=256, low_ch=48),
        "PSPNet": lambda n_in, n_cls, args=None: PSPNet(n_channels=n_in, n_classes=n_cls, feat=128, ppm_out=128, bins=(1, 2, 3, 6)),
        "TransUNet": lambda n_in, n_cls, args=None: TransUNet(n_channels=n_in, n_classes=n_cls, embed_dim=256, patch=16, vit_depth=4),
        "SegmenterMini": lambda n_in, n_cls, args=None: SegmenterMini(n_channels=n_in, n_classes=n_cls, embed_dim=256, patch=16, depth=6),
        "Mask2FormerMini": lambda n_in, n_cls, args=None: Mask2FormerMini(n_channels=n_in, n_classes=n_cls, dim=256, num_queries=100, dec_layers=3),
    }


# ---------- 从权重路径中解析 {dataset}_{Model}_last_epoch.pth ----------
CKPT_RE = re.compile(r"(?P<ds>[A-Za-z0-9]+)_(?P<arch>.+)_last_epoch\.pth$")


def infer_arch_from_ckpt(ckpt_path: str) -> str:
    name = os.path.basename(ckpt_path)
    m = CKPT_RE.match(name)
    if not m:
        raise ValueError(f"无法从权重文件名识别模型名：{name}")
    return m.group("arch")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_thop_profile(model: nn.Module, x: torch.Tensor) -> Dict[str, str]:
    """
    macs/flops 用 thop 计算（如果环境没有 thop 就返回 NA）
    """
    try:
        from thop import profile  # type: ignore
        model.eval()
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        # 常见约定：FLOPs ~ 2 * MACs（乘加算两次浮点运算）
        flops = 2.0 * float(macs)
        return {
            "macs": str(float(macs)),
            "flops": str(float(flops)),
            "params": str(float(params)),
        }
    except Exception:
        # thop 不存在或模型不兼容，就给 NA
        return {
            "macs": "NA",
            "flops": "NA",
            "params": "NA",
        }


def preprocess_to_tensor(img_pil: Image.Image, scale: float, device: torch.device) -> torch.Tensor:
    arr = BasicDataset.preprocess(None, img_pil, scale, is_mask=False)
    x = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)
    return x


def predict_tensor_to_mask(
    net: nn.Module,
    x: torch.Tensor,
    out_threshold: float,
    orig_hw: Optional[tuple] = None,
) -> np.ndarray:
    """
    x: [1, C, H, W]
    返回: [H, W] 的 numpy mask（long）
    """
    net.eval()
    with torch.no_grad():
        out = net(x)

        if isinstance(out, (tuple, list)):
            out = out[-1]

        if orig_hw is not None:
            H, W = orig_hw
            if out.shape[-2:] != (H, W):
                out = F.interpolate(out, (H, W), mode="bilinear", align_corners=False)

        # 多类：argmax；二类单通道：sigmoid+阈值
        n_classes = getattr(net, "n_classes", out.shape[1] if out.ndim == 4 else 2)
        if out.ndim == 4 and out.shape[1] > 1 and n_classes > 1:
            mask = out.argmax(dim=1)
        else:
            mask = (torch.sigmoid(out) > out_threshold).long().squeeze(1)

        return mask[0].long().cpu().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def profile_forward_time(
    net: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    """
    返回：平均 forward 时间（秒）
    """
    net.eval()
    # warmup
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / max(1, iters)


def get_args():
    p = argparse.ArgumentParser(description="Predict masks from input images (with optional profiling)")
    p.add_argument("--model", "-m", required=True, help="权重文件路径，如 dem4_ScConv_UNet_last_epoch.pth")
    p.add_argument("--input", "-i", nargs="+", required=True, help="输入图像文件名（可多个）")
    p.add_argument("--output", "-o", nargs="+", help="输出掩码文件名（可选，对应 input 一一匹配）")

    p.add_argument("--viz", "-v", action="store_true", help="窗口可视化")
    p.add_argument("--no-save", "-n", action="store_true", help="不保存输出 mask（减少 IO）")
    p.add_argument("--save-masks", action="store_true", help="强制保存 mask（用于 effecient.py 的探测）")

    p.add_argument("--mask-threshold", "-t", type=float, default=0.5, help="二值阈值（对单通道）")
    p.add_argument("--scale", "-s", type=float, default=0.5, help="输入缩放")
    p.add_argument("--bilinear", action="store_true", default=False, help="UNet 类是否用双线性上采样")
    p.add_argument("--classes", "-c", type=int, default=2, help="类别数（默认2）")
    p.add_argument("--in-channels", type=int, default=1, help="输入通道（默认1）")

    # ====== effecient.py 需要的 profile 参数 ======
    p.add_argument("--profile-warmup", type=int, default=0, help="profile warmup 次数")
    p.add_argument("--profile-iters", type=int, default=0, help="profile 计时迭代次数（>0 才启用 profile 输出）")

    return p.parse_args()


def get_output_filenames(args):
    if args.output:
        return args.output
    return [f"{os.path.splitext(fn)[0]}_OUT.png" for fn in args.input]


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save-masks 优先级高于 no-save
    if args.save_masks:
        args.no_save = False

    # 1) 解析模型名 & 构建网络
    arch = infer_arch_from_ckpt(args.model)
    builders = make_builders()
    if arch not in builders or builders[arch] is None:
        raise ValueError(f"未在 MODEL_BUILDERS 中找到模型：{arch}")

    net = builders[arch](args.in_channels, args.classes, args)
    if not hasattr(net, "n_classes"):
        net.n_classes = args.classes
    if not hasattr(net, "n_channels"):
        net.n_channels = args.in_channels
    net.to(device=device)

    logging.info(f"Loading model: {args.model} (arch={arch})")
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict, strict=False)
    logging.info("Model loaded!")

    out_files = get_output_filenames(args)

    # 用第一张图做 profile 的输入（effecient.py 每次就传一张 tile，本身就合理）
    prof_x = None
    prof_orig_hw = None

    for i, filename in enumerate(args.input):
        logging.info(f"Predicting image {filename} ...")
        img = Image.open(filename)
        prof_orig_hw = (img.size[1], img.size[0])  # (H, W)

        x = preprocess_to_tensor(img, args.scale, device)
        if prof_x is None:
            prof_x = x

        mask = predict_tensor_to_mask(net, x, args.mask_threshold, orig_hw=prof_orig_hw)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f"Mask saved to {out_filename}")

        if args.viz:
            logging.info("Visualizing results (close window to continue)...")
            plot_img_and_mask(img, mask)

    # ====== 打印 profile 行（effecient.py 解析用）=====
    # 只要 iters > 0 就启用
    if args.profile_iters and args.profile_iters > 0:
        if prof_x is None:
            # 理论不会发生，因为 input 必填
            dummy = torch.zeros((1, args.in_channels, 256, 256), device=device)
            prof_x = dummy

        avg_sec = profile_forward_time(
            net=net,
            x=prof_x,
            warmup=max(0, int(args.profile_warmup)),
            iters=max(1, int(args.profile_iters)),
            device=device,
        )

        # params：一定可算
        params_int = count_params(net)

        # macs/flops：尝试 thop
        th = try_thop_profile(net, prof_x)
        macs = th.get("macs", "NA")
        flops = th.get("flops", "NA")

        # 如果 thop 失败，就至少把 params 给出来
        if th.get("params", "NA") == "NA":
            params_out = str(params_int)
        else:
            # thop 的 params 可能是 float 字符串，这里优先用我们自己统计的 int
            params_out = str(params_int)

        # 重要：这一行必须稳定输出，effecient.py 才能抓到
        print(f"[PROFILE] params={params_out} macs={macs} flops={flops} avg_forward_sec={avg_sec}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
