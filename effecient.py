# -*- coding: utf-8 -*-
import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ========================== 可配置项 ==========================
BASE_DIR = Path(r"E:\Pytorch-UNet-master")  # 项目根目录
CHECKPOINT_ROOT = BASE_DIR / "checkpoints" / "autodl"
DATASETS = ["dem4"]

# predict.py 配置
PYTHON_EXE = "python"  # 或 python3
PREDICT_SCRIPT = BASE_DIR / "predict.py"
IN_CHANNELS = 3
NUM_CLASSES = 2
SCALE = 0.5
MASK_THRESHOLD = 0.5
BILINEAR = False

# 输入切片
TILE_IMAGE_EXT = ".jpg"  # 切片图像后缀
# ============================================================

def imgs_dir_of(name: str) -> Path:
    """切片图所在目录（输入）"""
    return BASE_DIR / "test_set" / name / "data_suda_seamount_bathymetry" / "data" / "imgs"

# -------- [PROFILE] 行解析 --------
PROFILE_RE_STRICT = re.compile(
    r"^\[PROFILE\]\s+params=(?P<params>\S+)\s+macs=(?P<macs>\S+)\s+flops=(?P<flops>\S+)\s+avg_forward_sec=(?P<avg_forward_sec>\S+)\s*$"
)
PROFILE_KV = re.compile(r"(\w+)=([^\s]+)")

def _extract_profile_line(stdout: str) -> Optional[str]:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("[PROFILE]"):
            return line
    return None

def parse_profile(stdout: str) -> Optional[Dict[str, str]]:
    """
    从 predict.py stdout 中提取 [PROFILE] 行
    - 先严格匹配（固定列）
    - 失败则做宽松 KV 解析
    """
    line = _extract_profile_line(stdout)
    if line is None:
        return None

    m = PROFILE_RE_STRICT.match(line)
    if m:
        return m.groupdict()

    kvs = dict(PROFILE_KV.findall(line))
    if kvs:
        kvs["raw"] = line
        return kvs

    return {"raw": line}

def parse_args():
    ap = argparse.ArgumentParser(description="批量统计模型计算效率（FLOPs/Params/forward time）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--debug", action="store_true", help="打印 predict.py 的 stdout/stderr（成功也打印）")
    ap.add_argument("--fail-fast", action="store_true", help="任意失败立即退出")
    ap.add_argument("--profile-warmup", type=int, default=10, help="profile warmup 次数")
    ap.add_argument("--profile-iters", type=int, default=50, help="profile 计时迭代次数")
    ap.add_argument("--profile-every-tile", action="store_true", help="对每张 tile 都跑一次 profile（非常慢，不推荐）")
    ap.add_argument("--csv", type=str, default="efficiency_summary.csv", help="输出 CSV 文件名（相对 BASE_DIR 或绝对路径）")
    ap.add_argument("--save-masks", action="store_true", help="是否让 predict.py 输出 mask（开启则传对应参数；默认不保存，减少 IO）")
    return ap.parse_args()

def _env_for_subprocess() -> Dict[str, str]:
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env

def _shorten(text: str, max_lines: int = 60) -> str:
    """Debug输出截断：最多 max_lines 行"""
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return (text or "").strip()
    head = "\n".join(lines[:max_lines])
    return head.strip() + "\n... (truncated)"

def _run_predict_help() -> str:
    """
    运行: python predict.py -h
    用来探测 predict.py 支持哪些参数，避免传不存在的 flag。
    """
    cmd = [PYTHON_EXE, str(PREDICT_SCRIPT), "-h"]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_env_for_subprocess(),
    )
    # argparse 的 help 一般在 stdout（偶尔在 stderr）
    return (r.stdout or "") + "\n" + (r.stderr or "")

def _detect_predict_caps() -> Dict[str, bool]:
    help_text = _run_predict_help()
    caps = {
        "has_model": ("--model" in help_text),
        "has_input": ("--input" in help_text),
        "has_profile_warmup": ("--profile-warmup" in help_text),
        "has_profile_iters": ("--profile-iters" in help_text),
        "has_save_masks": ("--save-masks" in help_text),
        "has_no_save": ("--no-save" in help_text) or ("-n" in help_text),
    }
    return caps

def run_profile(
    ckpt_path: Path,
    tile_path: Path,
    warmup: int,
    iters: int,
    dry_run: bool,
    debug: bool,
    save_masks: bool,
    caps: Dict[str, bool],
) -> Dict[str, str]:
    """
    调用 predict.py，返回一个 dict（包含错误信息时也返回，便于写 CSV）
    """
    cmd: List[str] = [
        PYTHON_EXE,
        str(PREDICT_SCRIPT),
        "--model", str(ckpt_path),
        "--input", str(tile_path),
        "--in-channels", str(IN_CHANNELS),
        "--classes", str(NUM_CLASSES),
        "--scale", str(SCALE),
        "--mask-threshold", str(MASK_THRESHOLD),
        "--profile-warmup", str(warmup),
        "--profile-iters", str(iters),
    ]
    if BILINEAR:
        cmd.append("--bilinear")

    # ✅ 只在 predict.py 真支持时才传，避免"参数不存在"炸掉
    if save_masks:
        if caps.get("has_save_masks", False):
            cmd.append("--save-masks")
        else:
            # 不支持就不传，但把信息带回去
            pass
    else:
        # 默认不保存 mask：如果支持 --no-save 就传，不支持就啥也不传
        if caps.get("has_no_save", False):
            cmd.append("--no-save")

    # ✅ 防止你又在别处塞进来
    if "--profile" in cmd:
        raise RuntimeError("BUG: cmd contains illegal flag --profile (must be removed)")

    print("THIS SCRIPT :", Path(__file__).resolve())
    print("PREDICT_SCRIPT:", PREDICT_SCRIPT.resolve())
    print("RUN:", " ".join(map(str, cmd)))

    if dry_run:
        return {"status": "DRY_RUN"}

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_env_for_subprocess(),
    )

    if debug:
        print("---- predict.py stdout (debug) ----")
        print(_shorten(r.stdout, max_lines=200))
        print("---- predict.py stderr (debug) ----")
        print(_shorten(r.stderr, max_lines=200))
        print("-----------------------------------")

    if r.returncode != 0:
        return {
            "status": "ERROR",
            "returncode": str(r.returncode),
            "error": "predict.py failed",
            "cmd": " ".join(map(str, cmd)),
            "stderr": _shorten(r.stderr, max_lines=120),
            "stdout": _shorten(r.stdout, max_lines=120),
        }

    prof = parse_profile(r.stdout or "")
    if prof is None:
        return {
            "status": "ERROR",
            "error": "missing_profile_line",
            "cmd": " ".join(map(str, cmd)),
            "stdout": _shorten(r.stdout, max_lines=200),
            "stderr": _shorten(r.stderr, max_lines=120),
        }

    prof_out = dict(prof)
    prof_out["status"] = "OK"
    return prof_out

def write_csv(rows: List[Dict[str, str]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "dataset",
        "model",
        "checkpoint",
        "tile",
        "status",
        "params",
        "macs",
        "flops",
        "avg_forward_sec",
        "raw",
        "returncode",
        "error",
        "cmd",
        "stdout",
        "stderr",
    ]
    seen = set()
    cols: List[str] = []
    for c in base_cols:
        cols.append(c)
        seen.add(c)

    for r in rows:
        for k in r.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = str(r.get(c, ""))
                v = v.replace('"', '""')
                if ("," in v) or ("\n" in v):
                    v = f'"{v}"'
                vals.append(v)
            f.write(",".join(vals) + "\n")
    print(f"[OK] Saved CSV: {out_csv}")

def main():
    args = parse_args()

    # 基本检查
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError(f"predict.py 不存在：{PREDICT_SCRIPT}")
    if not CHECKPOINT_ROOT.exists():
        raise FileNotFoundError(f"CHECKPOINT_ROOT 不存在：{CHECKPOINT_ROOT}")

    caps = _detect_predict_caps()
    # ✅ 关键：确保你调用到的 predict.py 真的是"推理脚本"
    if not (caps["has_model"] and caps["has_input"]):
        raise RuntimeError(
            "你当前的 PREDICT_SCRIPT 看起来不是用于推理的 predict.py（help 里没有 --model/--input）。\n"
            f"PREDICT_SCRIPT = {PREDICT_SCRIPT.resolve()}\n"
            "请检查是不是路径指错/文件被覆盖。"
        )
    if not (caps["has_profile_warmup"] and caps["has_profile_iters"]):
        raise RuntimeError(
            "你当前的 predict.py help 里没有 --profile-warmup/--profile-iters。\n"
            "说明这个 predict.py 版本不支持你现在的 profile 调用方式。"
        )

    out_csv = Path(args.csv)
    if not out_csv.is_absolute():
        out_csv = BASE_DIR / out_csv

    summary: List[Dict[str, str]] = []

    for ds in DATASETS:
        print(f"\n=== Dataset: {ds} ===")

        ckpts = sorted(CHECKPOINT_ROOT.rglob(f"{ds}_*_last_epoch.pth"))
        if not ckpts:
            print(f"[Warn] 未找到权重：{CHECKPOINT_ROOT}/**/{ds}_*_last_epoch.pth")
            continue

        imgs_dir = imgs_dir_of(ds)
        if not imgs_dir.exists():
            print(f"[Warn] 找不到切片目录：{imgs_dir}（跳过）")
            continue

        tiles = sorted([p for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() == TILE_IMAGE_EXT.lower()])
        if not tiles:
            print(f"[Warn] 切片为空：{imgs_dir}")
            continue

        print(f"[{ds}] tiles={len(tiles)} ckpts={len(ckpts)}")

        for ckpt in ckpts:
            model_name = ckpt.name[len(ds) + 1: -len("_last_epoch.pth")]
            print(f"\n--- Model: {model_name} | ckpt={ckpt.name} ---")

            chosen_tiles = tiles if args.profile_every_tile else [tiles[0]]
            for tile in chosen_tiles:
                prof = run_profile(
                    ckpt_path=ckpt,
                    tile_path=tile,
                    warmup=args.profile_warmup,
                    iters=args.profile_iters,
                    dry_run=args.dry_run,
                    debug=args.debug,
                    save_masks=args.save_masks,
                    caps=caps,
                )
                row = {
                    "dataset": ds,
                    "model": model_name,
                    "checkpoint": str(ckpt),
                    "tile": str(tile),
                }
                row.update(prof)
                summary.append(row)

                # 默认：每个模型只跑一次（无论 OK/ERROR 都不在这个模型上继续）
                if not args.profile_every_tile:
                    break

                # fail-fast
                if args.fail_fast and prof.get("status") != "OK":
                    if not args.dry_run:
                        write_csv(summary, out_csv)
                    raise SystemExit("[Fail-Fast] 遇到错误，已退出（并写出当前 CSV）")

            if args.fail_fast and summary and summary[-1].get("status") != "OK":
                # 保险：按模型粒度 fail-fast
                if not args.dry_run:
                    write_csv(summary, out_csv)
                raise SystemExit("[Fail-Fast] 遇到错误，已退出（并写出当前 CSV）")

    if not args.dry_run:
        write_csv(summary, out_csv)

if __name__ == "__main__":
    main()