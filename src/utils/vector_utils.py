"""
矢量数据（Shapefile）加载与检查工具
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


@dataclass
class VectorInfo:
    """矢量数据元信息"""
    path: str
    crs: str
    geometry_type: str
    feature_count: int
    bounds: tuple  # (minx, miny, maxx, maxy)
    columns: list
    has_valid_geometry: bool
    invalid_geometry_count: int
    area_stats: dict | None  # 面积统计（如果有面要素）


def load_vector(path: str | Path) -> gpd.GeoDataFrame:
    """
    加载矢量数据

    Args:
        path: Shapefile 文件路径

    Returns:
        GeoDataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"矢量文件不存在: {path}")

    gdf = gpd.read_file(path)
    return gdf


def get_vector_info(path: str | Path) -> VectorInfo:
    """
    获取矢量数据元信息

    Args:
        path: Shapefile 文件路径

    Returns:
        VectorInfo 数据类实例
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"矢量文件不存在: {path}")

    gdf = gpd.read_file(path)

    # 检查几何有效性
    valid_mask = gdf.geometry.is_valid
    invalid_count = int((~valid_mask).sum())
    has_valid = invalid_count == 0

    # 获取几何类型
    geom_types = gdf.geometry.geom_type.unique()
    geom_type_str = ", ".join(sorted(geom_types))

    # 面积统计（仅对 Polygon/MultiPolygon）
    area_stats = None
    if any(t in ["Polygon", "MultiPolygon"] for t in geom_types):
        # 先尝试计算地理坐标面积
        try:
            areas = gdf.geometry.area
            area_stats = {
                "count": len(areas),
                "min": float(areas.min()),
                "max": float(areas.max()),
                "mean": float(areas.mean()),
                "median": float(areas.median()),
                "std": float(areas.std()),
            }
        except Exception:
            area_stats = None

    return VectorInfo(
        path=str(path),
        crs=str(gdf.crs),
        geometry_type=geom_type_str,
        feature_count=len(gdf),
        bounds=tuple(gdf.total_bounds),
        columns=list(gdf.columns),
        has_valid_geometry=has_valid,
        invalid_geometry_count=invalid_count,
        area_stats=area_stats,
    )


def check_vector_geometry(gdf: gpd.GeoDataFrame) -> dict:
    """
    检查矢量几何有效性

    Args:
        gdf: GeoDataFrame

    Returns:
        检查结果字典
    """
    results = {
        "total": len(gdf),
        "valid": 0,
        "invalid": 0,
        "empty": 0,
        "details": [],
    }

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            results["empty"] += 1
            results["details"].append({
                "index": idx,
                "issue": "empty_geometry",
            })
        elif not geom.is_valid:
            results["invalid"] += 1
            results["details"].append({
                "index": idx,
                "issue": "invalid_geometry",
                "reason": geom.is_valid_reason() if hasattr(geom, 'is_valid_reason') else "unknown",
            })
        else:
            results["valid"] += 1

    return results


def fix_vector_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    修复无效几何

    Args:
        gdf: GeoDataFrame

    Returns:
        修复后的 GeoDataFrame
    """
    # 使用 buffer(0) 修复自相交等常见问题
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def print_vector_info(info: VectorInfo, name: str = "矢量") -> None:
    """打印矢量信息"""
    print(f"\n{'='*50}")
    print(f"{name} 信息")
    print(f"{'='*50}")
    print(f"文件路径: {info.path}")
    print(f"CRS: {info.crs}")
    print(f"几何类型: {info.geometry_type}")
    print(f"要素数量: {info.feature_count}")
    print(f"范围: {info.bounds}")
    print(f"字段: {info.columns}")
    print(f"几何有效: {info.has_valid_geometry}")
    if info.invalid_geometry_count > 0:
        print(f"无效几何数: {info.invalid_geometry_count}")
    if info.area_stats:
        print(f"面积统计:")
        print(f"  最小: {info.area_stats['min']:.6f}")
        print(f"  最大: {info.area_stats['max']:.6f}")
        print(f"  均值: {info.area_stats['mean']:.6f}")
        print(f"  中位: {info.area_stats['median']:.6f}")
    print(f"{'='*50}\n")
