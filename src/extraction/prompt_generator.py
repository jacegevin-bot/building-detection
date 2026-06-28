"""
提示生成模块
从训练矢量标注生成 SAM 提示（Box / Point / Mask）
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box


@dataclass
class PromptSet:
    """单个建筑物的提示集合"""
    building_id: int
    bbox: tuple  # (x1, y1, x2, y2) 像素坐标
    positive_points: np.ndarray  # (N, 2) 像素坐标
    negative_points: np.ndarray  # (M, 2) 像素坐标
    mask: np.ndarray | None  # 二值掩膜


@dataclass
class PromptBatch:
    """提示批次"""
    prompts: list[PromptSet]
    image_shape: tuple  # (height, width)
    prompt_strategy: str


def vector_to_pixel_coords(
    gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
) -> gpd.GeoDataFrame:
    """
    将矢量地理坐标转换为像素坐标

    Args:
        gdf: GeoDataFrame（地理坐标）
        transform: 影像仿射变换

    Returns:
        像素坐标的 GeoDataFrame
    """
    gdf_pixel = gdf.copy()
    # rasterio 的 transform 是 (地理坐标 -> 像素坐标)
    # 需要取逆：像素坐标 -> 地理坐标
    inv_transform = ~transform

    # 对每个几何进行坐标转换
    def transform_geom(geom):
        if geom.geom_type == "Polygon":
            exterior = [
                inv_transform * (x, y)
                for x, y in geom.exterior.coords
            ]
            interiors = []
            for interior in geom.interiors:
                interiors.append([
                    inv_transform * (x, y)
                    for x, y in interior.coords
                ])
            from shapely.geometry import Polygon
            return Polygon(exterior, interiors)
        elif geom.geom_type == "MultiPolygon":
            from shapely.geometry import MultiPolygon
            return MultiPolygon([transform_geom(p) for p in geom.geoms])
        return geom

    gdf_pixel["geometry"] = gdf_pixel.geometry.apply(transform_geom)
    return gdf_pixel


def generate_box_prompts(
    gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    image_shape: tuple,
) -> list[tuple]:
    """
    生成 Box 提示（外接矩形）

    Args:
        gdf: 建筑物矢量
        transform: 影像仿射变换
        image_shape: 影像尺寸 (height, width)

    Returns:
        box 列表 [(x1, y1, x2, y2), ...]
    """
    boxes = []
    inv_transform = ~transform

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 获取外接矩形
        minx, miny, maxx, maxy = geom.bounds

        # 转换为像素坐标
        px_minx, px_miny = inv_transform * (minx, maxy)  # 注意：地理坐标 y 轴向上
        px_maxx, px_maxy = inv_transform * (maxx, miny)

        # 裁剪到影像范围
        h, w = image_shape
        px_minx = max(0, int(px_minx))
        px_miny = max(0, int(px_miny))
        px_maxx = min(w, int(px_maxx))
        px_maxy = min(h, int(px_maxy))

        if px_minx < px_maxx and px_miny < px_maxy:
            boxes.append((px_minx, px_miny, px_maxx, px_maxy))

    return boxes


def generate_point_prompts(
    gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    image_shape: tuple,
    num_positive: int = 5,
    num_negative: int = 3,
    negative_distance: float = 50.0,
) -> list[tuple]:
    """
    生成 Point 提示（正负点）

    Args:
        gdf: 建筑物矢量
        transform: 影像仿射变换
        image_shape: 影像尺寸 (height, width)
        num_positive: 正点数量
        num_negative: 负点数量
        negative_distance: 负点距离建筑物的距离（像素）

    Returns:
        (positive_points, negative_points) 列表
    """
    inv_transform = ~transform
    h, w = image_shape
    results = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 生成正点（在建筑物内部）
        pos_points = []
        bounds = geom.bounds
        attempts = 0
        while len(pos_points) < num_positive and attempts < 100:
            # 在外接矩形内随机采样
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            from shapely.geometry import Point
            if geom.contains(Point(x, y)):
                px, py = inv_transform * (x, y)
                px, py = int(px), int(py)
                if 0 <= px < w and 0 <= py < h:
                    pos_points.append((px, py))
            attempts += 1

        # 如果采样不足，使用质心
        if len(pos_points) < num_positive:
            centroid = geom.centroid
            px, py = inv_transform * (centroid.x, centroid.y)
            px, py = int(px), int(py)
            if 0 <= px < w and 0 <= py < h:
                pos_points.append((px, py))

        # 生成负点（在建筑物外部，但在附近）
        neg_points = []
        buffered = geom.buffer(negative_distance)
        exterior = buffered.difference(geom) if not buffered.is_empty else None
        attempts = 0
        while len(neg_points) < num_negative and exterior and attempts < 100:
            minx, miny, maxx, maxy = buffered.bounds
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            from shapely.geometry import Point
            p = Point(x, y)
            if exterior.contains(p):
                px, py = inv_transform * (x, y)
                px, py = int(px), int(py)
                if 0 <= px < w and 0 <= py < h:
                    neg_points.append((px, py))
            attempts += 1

        results.append((
            np.array(pos_points) if pos_points else np.empty((0, 2)),
            np.array(neg_points) if neg_points else np.empty((0, 2)),
        ))

    return results


def generate_mask_prompts(
    gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    image_shape: tuple,
) -> list[np.ndarray]:
    """
    生成 Mask 提示（矢量栅格化）

    Args:
        gdf: 建筑物矢量
        transform: 影像仿射变换
        image_shape: 影像尺寸 (height, width)

    Returns:
        mask 列表
    """
    masks = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            masks.append(np.zeros(image_shape, dtype=np.uint8))
            continue

        # 栅格化单个建筑物
        mask = rasterize(
            [(geom, 1)],
            out_shape=image_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        masks.append(mask)

    return masks


def generate_prompts(
    gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    image_shape: tuple,
    strategy: str = "box_point_hybrid",
    num_positive: int = 5,
    num_negative: int = 3,
) -> PromptBatch:
    """
    生成提示批次

    Args:
        gdf: 建筑物矢量
        transform: 影像仿射变换
        image_shape: 影像尺寸 (height, width)
        strategy: 提示策略 (box_only / mask_only / box_point_hybrid)
        num_positive: 正点数量
        num_negative: 负点数量

    Returns:
        PromptBatch
    """
    prompts = []

    if strategy == "box_only":
        boxes = generate_box_prompts(gdf, transform, image_shape)
        for i, bbox in enumerate(boxes):
            prompts.append(PromptSet(
                building_id=i,
                bbox=bbox,
                positive_points=np.empty((0, 2)),
                negative_points=np.empty((0, 2)),
                mask=None,
            ))

    elif strategy == "mask_only":
        masks = generate_mask_prompts(gdf, transform, image_shape)
        for i, mask in enumerate(masks):
            prompts.append(PromptSet(
                building_id=i,
                bbox=None,
                positive_points=np.empty((0, 2)),
                negative_points=np.empty((0, 2)),
                mask=mask,
            ))

    elif strategy == "box_point_hybrid":
        boxes = generate_box_prompts(gdf, transform, image_shape)
        points = generate_point_prompts(
            gdf, transform, image_shape, num_positive, num_negative
        )
        for i, (bbox, (pos_pts, neg_pts)) in enumerate(zip(boxes, points)):
            prompts.append(PromptSet(
                building_id=i,
                bbox=bbox,
                positive_points=pos_pts,
                negative_points=neg_pts,
                mask=None,
            ))

    else:
        raise ValueError(f"未知的提示策略: {strategy}")

    return PromptBatch(
        prompts=prompts,
        image_shape=image_shape,
        prompt_strategy=strategy,
    )
