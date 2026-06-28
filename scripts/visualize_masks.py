#!/usr/bin/env python3
"""
掩膜可视化脚本 - 生成可查看的掩膜 PNG 图像

问题：原始 TIF 掩膜值为 0/1，打开后全黑看不到
解决：生成彩色叠加版本，便于查看
"""

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_mask_with_background(
    mask_path: str,
    image_path: str,
    output_path: str,
    mask_color: str = 'red',
    alpha: float = 0.5,
    title: str = 'Building Mask'
):
    """
    生成掩膜与影像叠加的可视化图

    Args:
        mask_path: 掩膜 TIF 路径
        image_path: 背景影像路径
        output_path: 输出 PNG 路径
        mask_color: 掩膜颜色
        alpha: 透明度
        title: 图标题
    """
    print(f"生成: {output_path}")

    # 读取掩膜
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # 读取背景影像
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：原始掩膜（热力图）
    ax1 = axes[0]
    im = ax1.imshow(mask, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f'{title} (Raw)', fontsize=12)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 右图：叠加在影像上
    ax2 = axes[1]
    ax2.imshow(image)

    # 创建叠加层
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    if mask_color == 'red':
        overlay[mask > 0] = [1, 0, 0, alpha]
    elif mask_color == 'blue':
        overlay[mask > 0] = [0, 0, 1, alpha]
    elif mask_color == 'green':
        overlay[mask > 0] = [0, 1, 0, alpha]
    elif mask_color == 'yellow':
        overlay[mask > 0] = [1, 1, 0, alpha]

    ax2.imshow(overlay)
    ax2.set_title(f'{title} (Overlay)', fontsize=12)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存")


def main():
    # 设置路径
    project_dir = Path("/mnt/d/遥感综合实习/遥感综合实习第三次实习task3/3_fewshot_building_change")
    masks_dir = project_dir / "results" / "masks"
    vis_dir = project_dir / "results" / "visualization"
    vis_dir.mkdir(exist_ok=True)

    # 背景影像
    test_image = str(project_dir / "Dataset" / "test" / "test_image_2023_independent_roof.tif")
    train_image = str(project_dir / "Dataset" / "train" / "train_image_2023_independent_roof.tif")

    # 1. 训练区伪标签
    visualize_mask_with_background(
        mask_path=str(masks_dir / "pseudo_label.tif"),
        image_path=train_image,
        output_path=str(vis_dir / "pseudo_label_visual.png"),
        mask_color='green',
        title='Training Area Pseudo Labels (999 buildings)'
    )

    # 2. 测试区 2023 建筑物
    visualize_mask_with_background(
        mask_path=str(masks_dir / "building_2023.tif"),
        image_path=test_image,
        output_path=str(vis_dir / "building_2023_visual.png"),
        mask_color='green',
        title='Test Area 2023 Buildings'
    )

    # 3. 测试区 2020 建筑物
    visualize_mask_with_background(
        mask_path=str(masks_dir / "building_2020.tif"),
        image_path=test_image,
        output_path=str(vis_dir / "building_2020_visual.png"),
        mask_color='green',
        title='Test Area 2020 Buildings'
    )

    # 4. 新增建筑物
    visualize_mask_with_background(
        mask_path=str(masks_dir / "change_added_raw.tif"),
        image_path=test_image,
        output_path=str(vis_dir / "change_added_visual.png"),
        mask_color='red',
        title='New Buildings (2023-2020)'
    )

    # 5. 拆除建筑物
    visualize_mask_with_background(
        mask_path=str(masks_dir / "change_removed_raw.tif"),
        image_path=test_image,
        output_path=str(vis_dir / "change_removed_visual.png"),
        mask_color='blue',
        title='Demolished Buildings (2020-2023)'
    )

    # 6. XOR 变化
    visualize_mask_with_background(
        mask_path=str(masks_dir / "change_xor_raw.tif"),
        image_path=test_image,
        output_path=str(vis_dir / "change_xor_visual.png"),
        mask_color='yellow',
        title='XOR Change Map'
    )

    print("\n✅ 所有可视化掩膜已生成！")
    print(f"输出目录: {vis_dir}")


if __name__ == "__main__":
    main()
