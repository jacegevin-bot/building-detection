"""
SAM2 模型封装模块
用于加载 SAM2-base 模型并进行推理
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class SAM2Config:
    """SAM2 模型配置"""
    model_type: str = "sam2_hiera_base"
    checkpoint: str = ""  # TODO: 设置权重路径
    device: str = "cuda"
    image_size: int = 1024
    # 提示参数
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.95


@dataclass
class SegmentationResult:
    """分割结果"""
    masks: np.ndarray  # (N, H, W) 二值掩膜
    scores: np.ndarray  # (N,) 置信度分数
    logits: np.ndarray  # (N, H, W) 原始 logits


class SAM2Wrapper:
    """
    SAM2 模型封装类

    使用方法:
        config = SAM2Config(checkpoint="path/to/sam2_checkpoint.pth")
        model = SAM2Wrapper(config)
        result = model.predict(image, box=[x1, y1, x2, y2])
    """

    def __init__(self, config: SAM2Config):
        """
        初始化 SAM2 模型

        Args:
            config: 模型配置
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None

    def load_model(self) -> None:
        """
        加载 SAM2 模型

        Raises:
            RuntimeError: 模型加载失败
            FileNotFoundError: 权重文件不存在
        """
        try:
            # 尝试导入 sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 检查权重文件
            checkpoint_path = Path(self.config.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM2 权重文件不存在: {checkpoint_path}\n"
                    f"请下载 SAM2-base 权重到 configs/ 目录\n"
                    f"下载地址: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pth"
                )

            # 构建模型
            model = build_sam2(
                config_file=self.config.model_type,
                ckpt_path=str(checkpoint_path),
                device=self.device,
            )

            # 创建预测器
            self.predictor = SAM2ImagePredictor(model)
            self.model = model

            print(f"✅ SAM2 模型加载成功")
            print(f"   模型类型: {self.config.model_type}")
            print(f"   设备: {self.device}")
            print(f"   权重: {checkpoint_path}")

        except ImportError as e:
            raise RuntimeError(
                "未安装 sam2 库。请执行以下命令安装:\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e

    def set_image(self, image: np.ndarray) -> None:
        """
        设置输入图像

        Args:
            image: RGB 图像 (H, W, 3)，uint8 格式

        Raises:
            RuntimeError: 模型未加载
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 确保图像是 RGB 格式
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # 转换为 PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        self.predictor.set_image(pil_image)

    def predict_with_box(
        self,
        box: list[int],
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        使用 Box 提示进行预测

        Args:
            box: 边界框 [x1, y1, x2, y2]
            multimask_output: 是否输出多个掩膜

        Returns:
            SegmentationResult
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")

        # 转换为 numpy 数组
        box_np = np.array(box)

        # 预测
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            multimask_output=multimask_output,
        )

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def predict_with_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        使用 Point 提示进行预测

        Args:
            points: 点坐标 (N, 2)，像素坐标
            labels: 点标签 (N,)，1=正点，0=负点
            multimask_output: 是否输出多个掩膜

        Returns:
            SegmentationResult
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")

        # 预测
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def predict_with_mask(
        self,
        mask: np.ndarray,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        使用 Mask 提示进行预测

        Args:
            mask: 输入掩膜 (H, W)，二值格式
            multimask_output: 是否输出多个掩膜

        Returns:
            SegmentationResult
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")

        # 确保 mask 形状为 (1, H, W)
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]

        # 预测
        masks, scores, logits = self.predictor.predict(
            mask_input=mask,
            multimask_output=multimask_output,
        )

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def predict_hybrid(
        self,
        box: list[int],
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> SegmentationResult:
        """
        使用混合提示进行预测（Box + Point）

        Args:
            box: 边界框 [x1, y1, x2, y2]
            points: 点坐标 (N, 2)，可选
            labels: 点标签 (N,)，可选
            multimask_output: 是否输出多个掩膜

        Returns:
            SegmentationResult
        """
        if self.predictor is None:
            raise RuntimeError("模型未加载")

        # 准备参数
        kwargs = {
            "box": np.array(box),
            "multimask_output": multimask_output,
        }

        if points is not None and len(points) > 0:
            kwargs["point_coords"] = points
            kwargs["point_labels"] = labels if labels is not None else np.ones(len(points))

        # 预测
        masks, scores, logits = self.predictor.predict(**kwargs)

        return SegmentationResult(
            masks=masks,
            scores=scores,
            logits=logits,
        )

    def predict_batch(
        self,
        prompts: list[dict],
        multimask_output: bool = False,
    ) -> list[SegmentationResult]:
        """
        批量预测

        Args:
            prompts: 提示列表，每个提示为字典:
                - box: [x1, y1, x2, y2]
                - points: (N, 2) 可选
                - labels: (N,) 可选
            multimask_output: 是否输出多个掩膜

        Returns:
            SegmentationResult 列表
        """
        results = []
        for prompt in prompts:
            if "box" in prompt:
                result = self.predict_hybrid(
                    box=prompt["box"],
                    points=prompt.get("points"),
                    labels=prompt.get("labels"),
                    multimask_output=multimask_output,
                )
                results.append(result)
        return results


def create_sam2_model(
    checkpoint: str = "",
    device: str = "cuda",
    model_type: str = "sam2_hiera_base",
) -> SAM2Wrapper:
    """
    创建 SAM2 模型实例

    Args:
        checkpoint: 权重文件路径
        device: 设备 (cuda/cpu)
        model_type: 模型类型

    Returns:
        SAM2Wrapper 实例
    """
    config = SAM2Config(
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
    )
    return SAM2Wrapper(config)
