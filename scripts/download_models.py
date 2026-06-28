#!/usr/bin/env python3
"""
模型下载脚本
下载 SAM2-base 和其他模型权重
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


# 模型下载链接
MODEL_URLS = {
    "sam2_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pth",
        "filename": "sam2_hiera_base_plus.pth",
        "size_mb": 300,
        "md5": "",  # TODO: 添加 MD5 校验
    },
    "sam2_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pth",
        "filename": "sam2_hiera_small.pth",
        "size_mb": 150,
        "md5": "",
    },
    "sam2_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pth",
        "filename": "sam2_hiera_tiny.pth",
        "size_mb": 100,
        "md5": "",
    },
}


def download_file(url: str, output_path: str, show_progress: bool = True) -> None:
    """
    下载文件

    Args:
        url: 下载链接
        output_path: 输出路径
        show_progress: 是否显示进度
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"下载: {url}")
    print(f"保存到: {output_path}")

    def progress_hook(count, block_size, total_size):
        if show_progress:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r下载进度: {percent}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(output_path), progress_hook)
        print(f"\n✅ 下载完成: {output_path}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def download_sam2_model(
    model_name: str = "sam2_hiera_base_plus",
    output_dir: str = "configs",
) -> str:
    """
    下载 SAM2 模型

    Args:
        model_name: 模型名称
        output_dir: 输出目录

    Returns:
        下载的文件路径
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"未知模型: {model_name}，可选: {list(MODEL_URLS.keys())}")

    model_info = MODEL_URLS[model_name]
    output_path = Path(output_dir) / model_info["filename"]

    # 检查是否已下载
    if output_path.exists():
        print(f"✅ 模型已存在: {output_path}")
        return str(output_path)

    # 下载
    download_file(model_info["url"], str(output_path))

    return str(output_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载模型权重")
    parser.add_argument(
        "--model",
        type=str,
        default="sam2_hiera_base_plus",
        choices=list(MODEL_URLS.keys()),
        help="模型名称",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs",
        help="输出目录",
    )

    args = parser.parse_args()

    try:
        path = download_sam2_model(args.model, args.output_dir)
        print(f"\n模型路径: {path}")
        print(f"请更新 configs/config.yaml 中的 model.primary.checkpoint 为上述路径")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
