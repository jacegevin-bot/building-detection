#!/usr/bin/env python3
"""
可视化脚本 - 生成实验报告所需的图件

生成内容：
1. 新增/拆除分布专题图
2. 典型区域叠加图
3. 变化与伪变化对比图
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar


def setup_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_raster(path: str) -> tuple:
    """加载栅格数据"""
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return data, transform, crs, bounds


def create_change_distribution_map(
    change_added_path: str,
    change_removed_path: str,
    test_image_path: str,
    output_path: str,
    title: str = "Building Change Distribution Map"
):
    """
    创建新增/拆除分布专题图

    Args:
        change_added_path: 新增掩膜路径
        change_removed_path: 拆除掩膜路径
        test_image_path: 测试影像路径
        output_path: 输出路径
        title: 图标题
    """
    print(f"生成分布专题图: {output_path}")

    # 加载数据
    added_mask, transform, crs, bounds = load_raster(change_added_path)
    removed_mask, _, _, _ = load_raster(change_removed_path)

    with rasterio.open(test_image_path) as src:
        test_image = src.read().transpose(1, 2, 0)

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 显示底图
    ax.imshow(test_image, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

    # 创建叠加掩膜
    overlay = np.zeros((*added_mask.shape, 4), dtype=np.float32)

    # 新增建筑物 - 红色
    added_pixels = added_mask > 0
    overlay[added_pixels] = [1, 0, 0, 0.6]  # 红色，60% 不透明度

    # 拆除建筑物 - 蓝色
    removed_pixels = removed_mask > 0
    overlay[removed_pixels] = [0, 0, 1, 0.6]  # 蓝色，60% 不透明度

    # 同时是新增和拆除 - 紫色
    both = added_pixels & removed_pixels
    overlay[both] = [1, 0, 1, 0.6]

    ax.imshow(overlay, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.6, label='New Buildings'),
        mpatches.Patch(facecolor='blue', alpha=0.6, label='Demolished Buildings'),
        mpatches.Patch(facecolor='purple', alpha=0.6, label='Changed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 添加比例尺
    try:
        scalebar = ScaleBar(1, location='lower right')
        ax.add_artist(scalebar)
    except:
        pass

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")


def create_typical_area_overlay(
    building_2023_path: str,
    building_2020_path: str,
    test_image_path: str,
    output_path: str,
    roi: tuple = None,
    title: str = "Typical Area Overlay"
):
    """
    创建典型区域叠加图

    Args:
        building_2023_path: 2023 建筑物掩膜路径
        building_2020_path: 2020 建筑物掩膜路径
        test_image_path: 测试影像路径
        output_path: 输出路径
        roi: 感兴趣区域 (left, bottom, right, top)，None 表示全图
        title: 图标题
    """
    print(f"生成典型区域叠加图: {output_path}")

    # 加载数据
    building_2023, transform, crs, bounds = load_raster(building_2023_path)
    building_2020, _, _, _ = load_raster(building_2020_path)

    with rasterio.open(test_image_path) as src:
        test_image = src.read().transpose(1, 2, 0)

    # 如果指定了 ROI，裁剪数据
    if roi is not None:
        left, bottom, right, top = roi
        # 转换为像素坐标
        col_start = int((left - bounds.left) / transform.a)
        col_end = int((right - bounds.left) / transform.a)
        row_start = int((bounds.top - top) / -transform.e)
        row_end = int((bounds.top - bottom) / -transform.e)

        # 裁剪
        test_image = test_image[row_start:row_end, col_start:col_end]
        building_2023 = building_2023[row_start:row_end, col_start:col_end]
        building_2020 = building_2020[row_start:row_end, col_start:col_end]

        # 更新边界
        new_bounds = (
            bounds.left + col_start * transform.a,
            bounds.top + row_end * transform.e,
            bounds.left + col_end * transform.a,
            bounds.top + row_start * transform.e,
        )
    else:
        new_bounds = bounds

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 2023 建筑物
    ax1 = axes[0]
    ax1.imshow(test_image, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])
    overlay_2023 = np.zeros((*building_2023.shape, 4), dtype=np.float32)
    overlay_2023[building_2023 > 0] = [0, 1, 0, 0.5]  # 绿色
    ax1.imshow(overlay_2023, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])
    ax1.set_title('2023 Buildings', fontsize=12)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # 2020 建筑物
    ax2 = axes[1]
    ax2.imshow(test_image, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])
    overlay_2020 = np.zeros((*building_2020.shape, 4), dtype=np.float32)
    overlay_2020[building_2020 > 0] = [0, 1, 0, 0.5]  # 绿色
    ax2.imshow(overlay_2020, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])
    ax2.set_title('2020 Buildings', fontsize=12)
    ax2.set_xlabel('Longitude')

    # 变化叠加
    ax3 = axes[2]
    ax3.imshow(test_image, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])

    overlay_change = np.zeros((*building_2023.shape, 4), dtype=np.float32)
    # 新增 (2023有，2020无)
    new_buildings = (building_2023 > 0) & (building_2020 == 0)
    overlay_change[new_buildings] = [1, 0, 0, 0.6]  # 红色

    # 拆除 (2020有，2023无)
    removed_buildings = (building_2020 > 0) & (building_2023 == 0)
    overlay_change[removed_buildings] = [0, 0, 1, 0.6]  # 蓝色

    # 未变化
    unchanged = (building_2023 > 0) & (building_2020 > 0)
    overlay_change[unchanged] = [0, 1, 0, 0.4]  # 绿色

    ax3.imshow(overlay_change, extent=[new_bounds.left, new_bounds.right, new_bounds.bottom, new_bounds.top])

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.6, label='Unchanged'),
        mpatches.Patch(facecolor='red', alpha=0.6, label='New'),
        mpatches.Patch(facecolor='blue', alpha=0.6, label='Demolished'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax3.set_title('Change Detection', fontsize=12)
    ax3.set_xlabel('Longitude')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")


def create_pseudo_change_comparison(
    change_xor_path: str,
    change_added_path: str,
    change_removed_path: str,
    output_path: str,
    title: str = "Change vs Pseudo-change Comparison"
):
    """
    创建变化与伪变化对比图

    Args:
        change_xor_path: XOR 变化掩膜路径
        change_added_path: 新增掩膜路径
        change_removed_path: 拆除掩膜路径
        output_path: 输出路径
        title: 图标题
    """
    print(f"生成变化与伪变化对比图: {output_path}")

    # 加载数据
    xor_mask, transform, crs, bounds = load_raster(change_xor_path)
    added_mask, _, _, _ = load_raster(change_added_path)
    removed_mask, _, _, _ = load_raster(change_removed_path)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XOR 变化掩膜
    ax1 = axes[0]
    im1 = ax1.imshow(xor_mask, cmap='hot', vmin=0, vmax=1)
    ax1.set_title('XOR Change Mask', fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 新增掩膜
    ax2 = axes[1]
    im2 = ax2.imshow(added_mask, cmap='Reds', vmin=0, vmax=1)
    ax2.set_title('New Buildings Mask', fontsize=12)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 拆除掩膜
    ax3 = axes[2]
    im3 = ax3.imshow(removed_mask, cmap='Blues', vmin=0, vmax=1)
    ax3.set_title('Demolished Buildings Mask', fontsize=12)
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--figure", choices=["all", "distribution", "overlay", "comparison"],
                       default="all", help="Which figure to generate")
    parser.add_argument("--roi", type=float, nargs=4, metavar=('left', 'bottom', 'right', 'top'),
                       help="Region of interest for overlay map")
    args = parser.parse_args()

    setup_chinese_font()

    # 设置路径
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    masks_dir = results_dir / "masks"

    if args.figure in ["all", "distribution"]:
        create_change_distribution_map(
            change_added_path=str(masks_dir / "change_added_raw.tif"),
            change_removed_path=str(masks_dir / "change_removed_raw.tif"),
            test_image_path="Dataset/test/test_image_2023_independent_roof.tif",
            output_path=str(figures_dir / "change_distribution_map.png"),
            title="Building Change Distribution Map (2020-2023)"
        )

    if args.figure in ["all", "overlay"]:
        create_typical_area_overlay(
            building_2023_path=str(masks_dir / "building_2023.tif"),
            building_2020_path=str(masks_dir / "building_2020.tif"),
            test_image_path="Dataset/test/test_image_2023_independent_roof.tif",
            output_path=str(figures_dir / "typical_area_overlay.png"),
            roi=args.roi,
            title="Typical Area: Building Change Overlay"
        )

    if args.figure in ["all", "comparison"]:
        create_pseudo_change_comparison(
            change_xor_path=str(masks_dir / "change_xor_raw.tif"),
            change_added_path=str(masks_dir / "change_added_raw.tif"),
            change_removed_path=str(masks_dir / "change_removed_raw.tif"),
            output_path=str(figures_dir / "change_comparison.png"),
            title="Change vs Pseudo-change Comparison"
        )

    print("\n✅ 所有图件生成完成！")
    print(f"输出目录: {figures_dir}")


if __name__ == "__main__":
    main()
