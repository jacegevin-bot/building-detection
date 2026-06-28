"""
栅格数据（GeoTIFF）加载与检查工具
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class RasterInfo:
    """栅格数据元信息"""
    path: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: CRS
    transform: Affine
    bounds: tuple  # (left, bottom, right, top)
    resolution: tuple  # (x_res, y_res)
    nodata: float | None
    valid_pixel_count: int
    total_pixel_count: int


def load_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    加载栅格数据

    Args:
        path: GeoTIFF 文件路径

    Returns:
        data: 影像数组 (bands, height, width) 或 (height, width)
        profile: 影像元信息字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"栅格文件不存在: {path}")

    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()

    return data, profile


def get_raster_info(path: str | Path) -> RasterInfo:
    """
    获取栅格数据元信息

    Args:
        path: GeoTIFF 文件路径

    Returns:
        RasterInfo 数据类实例
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"栅格文件不存在: {path}")

    with rasterio.open(path) as src:
        # 读取数据统计有效像元
        data = src.read()
        nodata = src.nodata

        # 计算有效像元数
        if nodata is not None:
            # 多波段：所有波段都有效的像元
            if data.ndim == 3:
                valid_mask = np.all(data != nodata, axis=0)
            else:
                valid_mask = data != nodata
            valid_count = int(np.sum(valid_mask))
        else:
            valid_count = data.size

        total_count = src.width * src.height

        # 计算分辨率
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

        return RasterInfo(
            path=str(path),
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            crs=src.crs,
            transform=src.transform,
            bounds=src.bounds,
            resolution=(res_x, res_y),
            nodata=nodata,
            valid_pixel_count=valid_count,
            total_pixel_count=total_count,
        )


def check_raster_consistency(
    raster1_info: RasterInfo,
    raster2_info: RasterInfo,
    name1: str = "影像1",
    name2: str = "影像2",
) -> dict:
    """
    检查两期栅格数据的一致性

    Args:
        raster1_info: 第一期栅格信息
        raster2_info: 第二期栅格信息
        name1: 第一期名称
        name2: 第二期名称

    Returns:
        检查结果字典
    """
    results = {
        "crs_match": raster1_info.crs == raster2_info.crs,
        "size_match": (
            raster1_info.width == raster2_info.width
            and raster1_info.height == raster2_info.height
        ),
        "transform_match": raster1_info.transform == raster2_info.transform,
        "resolution_match": raster1_info.resolution == raster2_info.resolution,
        "details": {},
    }

    # 详细信息
    results["details"]["crs"] = {
        name1: str(raster1_info.crs),
        name2: str(raster2_info.crs),
    }
    results["details"]["size"] = {
        name1: f"{raster1_info.width} x {raster1_info.height}",
        name2: f"{raster2_info.width} x {raster2_info.height}",
    }
    results["details"]["resolution"] = {
        name1: raster1_info.resolution,
        name2: raster2_info.resolution,
    }
    results["details"]["bounds"] = {
        name1: raster1_info.bounds,
        name2: raster2_info.bounds,
    }

    # 计算偏移量
    if not results["transform_match"]:
        dx = raster2_info.transform.c - raster1_info.transform.c
        dy = raster2_info.transform.f - raster1_info.transform.f
        results["details"]["offset_pixels"] = {
            "dx": dx / raster1_info.resolution[0],
            "dy": dy / raster1_info.resolution[1],
        }

    return results


def print_raster_info(info: RasterInfo, name: str = "栅格") -> None:
    """打印栅格信息"""
    print(f"\n{'='*50}")
    print(f"{name} 信息")
    print(f"{'='*50}")
    print(f"文件路径: {info.path}")
    print(f"尺寸: {info.width} x {info.height}")
    print(f"波段数: {info.bands}")
    print(f"数据类型: {info.dtype}")
    print(f"CRS: {info.crs}")
    print(f"分辨率: {info.resolution[0]:.8f} x {info.resolution[1]:.8f}")
    print(f"范围: {info.bounds}")
    print(f"NoData: {info.nodata}")
    print(f"有效像元: {info.valid_pixel_count} / {info.total_pixel_count}")
    print(f"{'='*50}\n")
