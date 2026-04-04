#!/usr/bin/env python3
"""
消融实验可视化 —— 横向拼接多个 debug 子文件夹的图片。

对于 result_dir 下的每个序列目录，将指定的多个子文件夹中的同名帧图片
横向拼接为一张大图，保存到与源文件夹并列的输出目录中。

输入列（从左到右）：
  1. Input（原始图片，从数据集目录读取）
  2. oar_depth_contact_debug
  3. oar_window_pre
  4. oar_window_post_pre_dep
  5. oar_window_post
  6. oar_reproj_error

用法:
  python lib/scripts/concat_ablation_vis.py \
      --result-dir results/promptbase_video_depthrefine_emdb2 \
      --dataset-root datasets/EMDB \
      --source-dirs oar_depth_contact_debug oar_window_pre \
                    oar_window_post_pre_dep oar_window_post oar_reproj_error \
      --labels "Input" "Depth+Contact" "Pre-OAR" "Post-PreDep" "Post-OAR" "Reproj Error" \
      --output-dir oar_ablation_concat \
      --target-height 480 \
      --num-workers 8
"""

import argparse
import os
import sys
import glob
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Tuple

import cv2
import numpy as np


def find_dataset_image_dir(dataset_root: str, seq_name: str) -> Optional[str]:
    """
    在 EMDB 数据集中查找序列对应的 images 目录。
    EMDB 结构: datasets/EMDB/P{n}/{seq_name}/images/
    """
    if not dataset_root or not os.path.isdir(dataset_root):
        return None

    # 遍历 P0, P1, ... 子目录
    for entry in sorted(os.listdir(dataset_root)):
        candidate = os.path.join(dataset_root, entry, seq_name, "images")
        if os.path.isdir(candidate):
            return candidate
    return None


def add_label_bar(
    img: np.ndarray,
    labels: List[str],
    col_widths: List[int],
    bar_height: int = 36,
    font_scale: float = 0.7,
    font_thickness: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """在拼接图顶部添加标签栏。"""
    W = img.shape[1]
    bar = np.full((bar_height, W, 3), bg_color, dtype=np.uint8)

    x_offset = 0
    for label, cw in zip(labels, col_widths):
        # 居中绘制文字
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        tx = x_offset + max(0, (cw - tw) // 2)
        ty = (bar_height + th) // 2
        cv2.putText(bar, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thickness, cv2.LINE_AA)
        x_offset += cw

    return np.vstack([bar, img])


def process_single_frame(
    frame_name: str,
    seq_dir: str,
    input_image_dir: Optional[str],
    source_dirs: List[str],
    output_dir: str,
    target_height: Optional[int],
    labels: Optional[List[str]],
    add_labels: bool,
    separator_width: int,
    label_bar_height: int,
) -> bool:
    """处理单帧：读取各子文件夹同名图片 → 缩放 → 横向拼接 → 保存。"""
    images = []
    col_widths = []

    # 1. Input 图片
    if input_image_dir is not None:
        input_path = os.path.join(input_image_dir, frame_name)
        if os.path.isfile(input_path):
            img = cv2.imread(input_path)
        else:
            # 尝试 png 格式
            input_path_png = os.path.splitext(input_path)[0] + ".png"
            if os.path.isfile(input_path_png):
                img = cv2.imread(input_path_png)
            else:
                img = None
    else:
        img = None

    if img is not None:
        images.append(img)
    else:
        # 用占位灰图
        images.append(None)

    # 2. 各 source_dirs 的图片
    for sd in source_dirs:
        src_path = os.path.join(seq_dir, sd, frame_name)
        if os.path.isfile(src_path):
            img = cv2.imread(src_path)
            images.append(img)
        else:
            images.append(None)

    # 至少需要一张有效图片确定尺寸
    valid_imgs = [im for im in images if im is not None]
    if not valid_imgs:
        return False

    # 确定目标高度
    if target_height is None:
        target_height = valid_imgs[0].shape[0]

    # 缩放所有图片到统一高度
    resized = []
    for im in images:
        if im is None:
            # 用第一张有效图的宽高比创建灰色占位图
            ref = valid_imgs[0]
            ratio = target_height / ref.shape[0]
            placeholder_w = int(ref.shape[1] * ratio)
            placeholder = np.full((target_height, placeholder_w, 3), 128, dtype=np.uint8)
            cv2.putText(placeholder, "N/A", (placeholder_w // 2 - 20, target_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            resized.append(placeholder)
        else:
            ratio = target_height / im.shape[0]
            new_w = int(im.shape[1] * ratio)
            resized.append(cv2.resize(im, (new_w, target_height), interpolation=cv2.INTER_AREA))

    # 记录各列宽度
    col_widths = [r.shape[1] for r in resized]

    # 添加分割线
    if separator_width > 0:
        sep = np.full((target_height, separator_width, 3), 255, dtype=np.uint8)
        parts = []
        for i, r in enumerate(resized):
            parts.append(r)
            if i < len(resized) - 1:
                parts.append(sep)
        concat = np.hstack(parts)
        # 更新 col_widths 以包含分割线
        total_col_widths = []
        for i, cw in enumerate(col_widths):
            total_col_widths.append(cw + (separator_width if i < len(col_widths) - 1 else 0))
        col_widths = total_col_widths
    else:
        concat = np.hstack(resized)

    # 添加标签栏
    if add_labels and labels is not None:
        concat = add_label_bar(concat, labels, col_widths, bar_height=label_bar_height)

    # 保存
    out_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(out_path, concat)
    return True


def process_sequence(
    seq_name: str,
    result_dir: str,
    dataset_root: str,
    source_dirs: List[str],
    output_subdir: str,
    target_height: Optional[int],
    labels: Optional[List[str]],
    add_labels: bool,
    separator_width: int,
    label_bar_height: int,
    num_workers: int,
    frame_stride: Optional[int],
) -> int:
    """处理一个序列。"""
    seq_dir = os.path.join(result_dir, seq_name)
    if not os.path.isdir(seq_dir):
        print(f"  [SKIP] {seq_name}: 目录不存在")
        return 0

    # 检查至少有一个 source_dir 存在
    existing_source_dirs = [sd for sd in source_dirs if os.path.isdir(os.path.join(seq_dir, sd))]
    if not existing_source_dirs:
        print(f"  [SKIP] {seq_name}: 没有找到任何 source 子目录")
        return 0

    # 收集所有帧名（取各文件夹的并集中的交集）
    frame_sets = []
    for sd in existing_source_dirs:
        sd_path = os.path.join(seq_dir, sd)
        frames = {f for f in os.listdir(sd_path) if f.endswith(('.jpg', '.png'))}
        frame_sets.append(frames)

    if not frame_sets:
        return 0

    # 取交集：只处理所有文件夹都有的帧
    common_frames = sorted(frame_sets[0].intersection(*frame_sets[1:]) if len(frame_sets) > 1 else frame_sets[0])

    # 帧采样
    if frame_stride is not None and frame_stride > 1:
        # 从文件名中提取帧号并按 stride 过滤
        filtered = []
        for fn in common_frames:
            try:
                frame_num = int(os.path.splitext(fn)[0])
                if frame_num % frame_stride == 0:
                    filtered.append(fn)
            except ValueError:
                filtered.append(fn)
        common_frames = filtered

    if not common_frames:
        print(f"  [SKIP] {seq_name}: 没有公共帧")
        return 0

    # 输入图片目录
    input_image_dir = find_dataset_image_dir(dataset_root, seq_name)
    if input_image_dir is None:
        print(f"  [WARN] {seq_name}: 未找到 Input 图片目录，将用占位图")

    # 创建输出目录
    output_dir = os.path.join(seq_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 构建 labels 列表（Input + source_dirs）
    if labels is None:
        all_labels = ["Input"] + source_dirs
    else:
        all_labels = labels

    # 处理函数
    worker_fn = partial(
        process_single_frame,
        seq_dir=seq_dir,
        input_image_dir=input_image_dir,
        source_dirs=source_dirs,
        output_dir=output_dir,
        target_height=target_height,
        labels=all_labels,
        add_labels=add_labels,
        separator_width=separator_width,
        label_bar_height=label_bar_height,
    )

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(worker_fn, common_frames)
        count = sum(results)
    else:
        count = 0
        for fn in common_frames:
            if worker_fn(fn):
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="消融实验可视化：横向拼接多个 debug 子文件夹的图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--result-dir", type=str, required=True,
                        help="结果根目录，如 results/promptbase_video_depthrefine_emdb2")
    parser.add_argument("--dataset-root", type=str, default="datasets/EMDB",
                        help="EMDB 数据集根目录（用于读取 Input 原始图片）")
    parser.add_argument("--source-dirs", nargs="+", required=True,
                        help="要拼接的子文件夹名称列表（不含 Input）")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="各列标签（包括 Input 在内），数量应为 len(source_dirs)+1。"
                             "若不指定则使用 ['Input'] + source_dirs")
    parser.add_argument("--output-dir", type=str, default="oar_ablation_concat",
                        help="输出子目录名称（在每个 seq_dir 下创建）")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="指定要处理的序列名列表。若不指定则处理 result-dir 下所有序列")
    parser.add_argument("--target-height", type=int, default=None,
                        help="统一缩放到的目标高度（像素）。若不指定则使用原始高度")
    parser.add_argument("--separator-width", type=int, default=2,
                        help="图片间分割线宽度（像素），0 表示无分割线")
    parser.add_argument("--no-labels", action="store_true",
                        help="不添加标签栏")
    parser.add_argument("--label-bar-height", type=int, default=36,
                        help="标签栏高度（像素）")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="帧采样步长（仅处理帧号为该值倍数的帧）。"
                             "若不指定则处理所有帧")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="并行处理的进程数")

    args = parser.parse_args()

    # 验证 labels 数量
    if args.labels is not None:
        expected = len(args.source_dirs) + 1  # +1 for Input
        if len(args.labels) != expected:
            print(f"[ERROR] labels 数量 ({len(args.labels)}) 应等于 source_dirs 数量 + 1 ({expected})")
            sys.exit(1)

    # 获取序列列表
    if args.sequences:
        seq_list = args.sequences
    else:
        # 自动检测 result_dir 下的所有序列目录
        seq_list = sorted([
            d for d in os.listdir(args.result_dir)
            if os.path.isdir(os.path.join(args.result_dir, d))
            and d != "intermediate"  # 排除中间结果目录
        ])

    if not seq_list:
        print(f"[ERROR] 在 {args.result_dir} 下没有找到序列目录")
        sys.exit(1)

    print("=" * 60)
    print("消融实验可视化 —— 横向拼接")
    print("=" * 60)
    print(f"Result dir:    {args.result_dir}")
    print(f"Dataset root:  {args.dataset_root}")
    print(f"Source dirs:   {args.source_dirs}")
    print(f"Labels:        {args.labels or (['Input'] + args.source_dirs)}")
    print(f"Output subdir: {args.output_dir}")
    print(f"Target height: {args.target_height or 'original'}")
    print(f"Separator:     {args.separator_width}px")
    print(f"Label bar:     {'OFF' if args.no_labels else f'{args.label_bar_height}px'}")
    print(f"Frame stride:  {args.frame_stride or 'all'}")
    print(f"Workers:       {args.num_workers}")
    print(f"Sequences:     {len(seq_list)}")
    print("=" * 60)

    total_count = 0
    for i, seq_name in enumerate(seq_list, 1):
        print(f"\n[{i}/{len(seq_list)}] Processing: {seq_name}")
        count = process_sequence(
            seq_name=seq_name,
            result_dir=args.result_dir,
            dataset_root=args.dataset_root,
            source_dirs=args.source_dirs,
            output_subdir=args.output_dir,
            target_height=args.target_height,
            labels=args.labels,
            add_labels=not args.no_labels,
            separator_width=args.separator_width,
            label_bar_height=args.label_bar_height,
            num_workers=args.num_workers,
            frame_stride=args.frame_stride,
        )
        print(f"  -> 保存 {count} 张拼接图到 {os.path.join(args.result_dir, seq_name, args.output_dir)}")
        total_count += count

    print(f"\n{'=' * 60}")
    print(f"完成！共处理 {total_count} 张图片")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
