"""
变化检测模块
计算两期建筑物掩膜之间的变化（新增/拆除）
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass
class ChangeDetectionResult:
    """变化检测结果"""
    added_mask: np.ndarray  # 新增掩膜 (H, W)
    removed_mask: np.ndarray  # 拆除掩膜 (H, W)
    xor_mask: np.ndarray  # XOR 变化掩膜 (H, W)
    added_count: int  # 新增像元数
    removed_count: int  # 拆除像元数
    xor_count: int  # 总变化像元数


class ChangeDetector:
    """
    变化检测器

    计算两期建筑物掩膜之间的变化:
    - 新增: building_2023 == 1 AND building_2020 == 0
    - 拆除: building_2020 == 1 AND building_2023 == 0
    - XOR: XOR(building_2020, building_2023)
    """

    def __init__(self):
        pass

    def detect(
        self,
        mask_2020: np.ndarray,
        mask_2023: np.ndarray,
    ) -> ChangeDetectionResult:
        """
        执行变化检测

        Args:
            mask_2020: 2020 年建筑物掩膜 (H, W)，1=建筑，0=非建筑
            mask_2023: 2023 年建筑物掩膜 (H, W)，1=建筑，0=非建筑

        Returns:
            ChangeDetectionResult

        Raises:
            ValueError: 掩膜尺寸不一致
        """
        # 检查尺寸
        if mask_2020.shape != mask_2023.shape:
            raise ValueError(
                f"掩膜尺寸不一致: {mask_2020.shape} vs {mask_2023.shape}"
            )

        # 确保二值化
        mask_2020 = (mask_2020 > 0).astype(np.uint8)
        mask_2023 = (mask_2023 > 0).astype(np.uint8)

        # 计算变化
        # 新增: 2023=1 AND 2020=0
        added = ((mask_2023 == 1) & (mask_2020 == 0)).astype(np.uint8)

        # 拆除: 2020=1 AND 2023=0
        removed = ((mask_2020 == 1) & (mask_2023 == 0)).astype(np.uint8)

        # XOR: 所有变化
        xor = (added | removed).astype(np.uint8)

        return ChangeDetectionResult(
            added_mask=added,
            removed_mask=removed,
            xor_mask=xor,
            added_count=int(np.sum(added)),
            removed_count=int(np.sum(removed)),
            xor_count=int(np.sum(xor)),
        )

    def detect_from_files(
        self,
        mask_2020_path: str,
        mask_2023_path: str,
        output_dir: str,
    ) -> ChangeDetectionResult:
        """
        从文件执行变化检测并保存结果

        Args:
            mask_2020_path: 2020 年掩膜路径
            mask_2023_path: 2023 年掩膜路径
            output_dir: 输出目录

        Returns:
            ChangeDetectionResult
        """
        # 读取掩膜
        with rasterio.open(mask_2020_path) as src:
            mask_2020 = src.read(1)
            profile = src.profile.copy()

        with rasterio.open(mask_2023_path) as src:
            mask_2023 = src.read(1)

        # 执行变化检测
        result = self.detect(mask_2020, mask_2023)

        # 保存结果
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_raster(
            result.added_mask,
            output_dir / "change_added_raw.tif",
            profile,
        )
        self._save_raster(
            result.removed_mask,
            output_dir / "change_removed_raw.tif",
            profile,
        )
        self._save_raster(
            result.xor_mask,
            output_dir / "change_xor_raw.tif",
            profile,
        )

        print(f"✅ 变化检测完成")
        print(f"   新增像元: {result.added_count}")
        print(f"   拆除像元: {result.removed_count}")
        print(f"   总变化像元: {result.xor_count}")

        return result

    def _save_raster(
        self,
        data: np.ndarray,
        output_path: Path,
        profile: dict,
    ) -> None:
        """
        保存栅格文件

        Args:
            data: 栅格数据 (H, W)
            output_path: 输出路径
            profile: 影像 profile
        """
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)


def run_change_detection(
    mask_2020_path: str,
    mask_2023_path: str,
    output_dir: str,
) -> ChangeDetectionResult:
    """
    运行变化检测

    Args:
        mask_2020_path: 2020 年掩膜路径
        mask_2023_path: 2023 年掩膜路径
        output_dir: 输出目录

    Returns:
        ChangeDetectionResult
    """
    detector = ChangeDetector()
    return detector.detect_from_files(mask_2020_path, mask_2023_path, output_dir)
