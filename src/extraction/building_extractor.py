"""
建筑物提取模块
集成模型和提示生成，完成建筑物掩膜提取
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine

from src.extraction.prompt_generator import (
    PromptBatch,
    PromptSet,
    generate_prompts,
)
from src.model.sam2_wrapper import SAM2Wrapper, SegmentationResult


@dataclass
class ExtractionConfig:
    """提取配置"""
    prompt_strategy: str = "box_point_hybrid"
    num_positive_points: int = 5
    num_negative_points: int = 3
    confidence_threshold: float = 0.5
    use_best_mask: bool = True  # 是否只使用最高分掩膜


@dataclass
class ExtractionResult:
    """提取结果"""
    mask: np.ndarray  # (H, W) 最终二值掩膜
    confidence: np.ndarray  # (H, W) 置信度图
    building_count: int  # 检测到的建筑物数量


class BuildingExtractor:
    """
    建筑物提取器

    使用方法:
        extractor = BuildingExtractor(model, config)
        result = extractor.extract(image, labels_gdf, transform)
    """

    def __init__(
        self,
        model: SAM2Wrapper,
        config: Optional[ExtractionConfig] = None,
    ):
        """
        初始化提取器

        Args:
            model: SAM2 模型
            config: 提取配置
        """
        self.model = model
        self.config = config or ExtractionConfig()

    def extract_single_building(
        self,
        image: np.ndarray,
        prompt: PromptSet,
    ) -> SegmentationResult:
        """
        提取单个建筑物

        Args:
            image: RGB 图像 (H, W, 3)
            prompt: 单个建筑物的提示

        Returns:
            SegmentationResult
        """
        # 设置图像
        self.model.set_image(image)

        # 根据提示类型选择预测方法
        if prompt.bbox is not None and len(prompt.positive_points) > 0:
            # 混合提示：Box + Points
            result = self.model.predict_hybrid(
                box=prompt.bbox,
                points=prompt.positive_points,
                labels=np.ones(len(prompt.positive_points)),
                multimask_output=False,
            )
        elif prompt.bbox is not None:
            # 纯 Box 提示
            result = self.model.predict_with_box(
                box=prompt.bbox,
                multimask_output=False,
            )
        elif prompt.mask is not None:
            # Mask 提示
            result = self.model.predict_with_mask(
                mask=prompt.mask,
                multimask_output=False,
            )
        else:
            raise ValueError("提示中缺少有效的输入")

        return result

    def extract_all_buildings(
        self,
        image: np.ndarray,
        prompts: PromptBatch,
    ) -> ExtractionResult:
        """
        提取所有建筑物

        Args:
            image: RGB 图像 (H, W, 3)
            prompts: 提示批次

        Returns:
            ExtractionResult
        """
        h, w = prompts.image_shape
        final_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        building_count = 0

        for i, prompt in enumerate(prompts.prompts):
            try:
                # 提取单个建筑物
                result = self.extract_single_building(image, prompt)

                # 选择最佳掩膜
                if self.config.use_best_mask:
                    best_idx = np.argmax(result.scores)
                    mask = result.masks[best_idx]
                    score = result.scores[best_idx]
                else:
                    mask = result.masks[0]
                    score = result.scores[0]

                # 应用置信度阈值
                if score >= self.config.confidence_threshold:
                    # 合并到最终掩膜
                    final_mask = np.maximum(final_mask, mask.astype(np.uint8))

                    # 更新置信度图（取最大值）
                    confidence_map = np.maximum(confidence_map, mask * score)

                    building_count += 1

                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1}/{len(prompts.prompts)} 个建筑物")

            except Exception as e:
                print(f"  ⚠️ 建筑物 {i} 提取失败: {e}")
                continue

        return ExtractionResult(
            mask=final_mask,
            confidence=confidence_map,
            building_count=building_count,
        )

    def extract_from_image(
        self,
        image_path: str,
        labels_path: str,
        output_path: str,
    ) -> ExtractionResult:
        """
        从影像文件提取建筑物并保存

        Args:
            image_path: 影像文件路径
            labels_path: 标注矢量路径
            output_path: 输出掩膜路径

        Returns:
            ExtractionResult
        """
        # 读取影像
        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0)  # (H, W, C)
            transform = src.transform
            profile = src.profile.copy()

        # 读取标注
        labels_gdf = gpd.read_file(labels_path)

        # 生成提示
        print(f"生成提示（策略: {self.config.prompt_strategy}）...")
        prompts = generate_prompts(
            gdf=labels_gdf,
            transform=transform,
            image_shape=(profile["height"], profile["width"]),
            strategy=self.config.prompt_strategy,
            num_positive=self.config.num_positive_points,
            num_negative=self.config.num_negative_points,
        )
        print(f"共生成 {len(prompts.prompts)} 个提示")

        # 提取建筑物
        print("开始提取建筑物...")
        result = self.extract_all_buildings(image, prompts)
        print(f"✅ 提取完成，检测到 {result.building_count} 个建筑物")

        # 保存结果
        self.save_mask(result.mask, output_path, profile)

        return result

    def save_mask(
        self,
        mask: np.ndarray,
        output_path: str,
        profile: dict,
    ) -> None:
        """
        保存掩膜到 GeoTIFF

        Args:
            mask: 二值掩膜 (H, W)
            output_path: 输出路径
            profile: 影像 profile
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 更新 profile
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

        print(f"✅ 掩膜已保存: {output_path}")


def run_extraction(
    model: SAM2Wrapper,
    image_path: str,
    labels_path: str,
    output_path: str,
    config: Optional[ExtractionConfig] = None,
) -> ExtractionResult:
    """
    运行建筑物提取

    Args:
        model: SAM2 模型
        image_path: 影像路径
        labels_path: 标注路径
        output_path: 输出路径
        config: 提取配置

    Returns:
        ExtractionResult
    """
    extractor = BuildingExtractor(model, config)
    return extractor.extract_from_image(image_path, labels_path, output_path)
