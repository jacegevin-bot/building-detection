"""
数据检查模块
用于验证训练/测试数据的完整性和一致性
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.raster_utils import (
    RasterInfo,
    get_raster_info,
    check_raster_consistency,
    print_raster_info,
)
from src.utils.vector_utils import (
    VectorInfo,
    get_vector_info,
    check_vector_geometry,
    print_vector_info,
    load_vector,
)


@dataclass
class DataCheckResult:
    """数据检查结果"""
    train_image: RasterInfo
    train_labels: VectorInfo
    train_aoi: VectorInfo
    test_image_2023: RasterInfo
    test_image_2020: RasterInfo
    test_aoi: VectorInfo
    registration_check: dict
    geometry_check: dict
    all_passed: bool
    issues: list


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_train_data(config: dict) -> tuple[RasterInfo, VectorInfo, VectorInfo, list]:
    """
    检查训练数据

    Args:
        config: 配置字典

    Returns:
        train_image_info, train_labels_info, train_aoi_info, issues
    """
    issues = []

    # 检查训练影像
    train_image_path = config["data"]["train"]["image"]
    print(f"检查训练影像: {train_image_path}")
    train_image_info = get_raster_info(train_image_path)
    print_raster_info(train_image_info, "训练影像")

    # 检查波段数
    if train_image_info.bands != 3:
        issues.append(f"训练影像波段数异常: {train_image_info.bands} (期望 3)")

    # 检查数据类型
    if train_image_info.dtype != "uint8":
        issues.append(f"训练影像数据类型异常: {train_image_info.dtype} (期望 uint8)")

    # 检查训练标注
    train_labels_path = config["data"]["train"]["labels"]
    print(f"检查训练标注: {train_labels_path}")
    train_labels_info = get_vector_info(train_labels_path)
    print_vector_info(train_labels_info, "训练标注")

    # 检查要素数量
    if train_labels_info.feature_count < 100:
        issues.append(f"训练标注要素数量偏少: {train_labels_info.feature_count}")

    # 检查几何有效性
    if not train_labels_info.has_valid_geometry:
        issues.append(f"训练标注存在无效几何: {train_labels_info.invalid_geometry_count} 个")

    # 检查训练 AOI
    train_aoi_path = config["data"]["train"]["aoi"]
    print(f"检查训练 AOI: {train_aoi_path}")
    train_aoi_info = get_vector_info(train_aoi_path)
    print_vector_info(train_aoi_info, "训练 AOI")

    return train_image_info, train_labels_info, train_aoi_info, issues


def check_test_data(config: dict) -> tuple[RasterInfo, RasterInfo, VectorInfo, dict, list]:
    """
    检查测试数据

    Args:
        config: 配置字典

    Returns:
        test_image_2023_info, test_image_2020_info, test_aoi_info, registration_check, issues
    """
    issues = []

    # 检查 2023 年测试影像
    test_image_2023_path = config["data"]["test"]["image_2023"]
    print(f"检查测试影像 2023: {test_image_2023_path}")
    test_image_2023_info = get_raster_info(test_image_2023_path)
    print_raster_info(test_image_2023_info, "测试影像 2023")

    # 检查 2020 年测试影像
    test_image_2020_path = config["data"]["test"]["image_2020"]
    print(f"检查测试影像 2020: {test_image_2020_path}")
    test_image_2020_info = get_raster_info(test_image_2020_path)
    print_raster_info(test_image_2020_info, "测试影像 2020")

    # 检查两期影像一致性
    print("检查两期影像一致性...")
    registration_check = check_raster_consistency(
        test_image_2023_info,
        test_image_2020_info,
        "2023",
        "2020",
    )

    if not registration_check["crs_match"]:
        issues.append("两期影像 CRS 不一致")
    if not registration_check["size_match"]:
        issues.append("两期影像尺寸不一致")
    if not registration_check["resolution_match"]:
        issues.append("两期影像分辨率不一致")
    if not registration_check["transform_match"]:
        offset = registration_check["details"].get("offset_pixels", {})
        dx = offset.get("dx", 0)
        dy = offset.get("dy", 0)
        # 忽略微小偏移（小于 0.01 像素，通常为浮点精度误差）
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            issues.append(f"两期影像存在偏移: dx={dx:.2f}px, dy={dy:.2f}px")
        else:
            print(f"  偏移量极小（dx={dx:.2e}px, dy={dy:.2e}px），可忽略")

    # 检查测试 AOI
    test_aoi_path = config["data"]["test"]["aoi"]
    print(f"检查测试 AOI: {test_aoi_path}")
    test_aoi_info = get_vector_info(test_aoi_path)
    print_vector_info(test_aoi_info, "测试 AOI")

    return test_image_2023_info, test_image_2020_info, test_aoi_info, registration_check, issues


def check_label_geometry(config: dict) -> dict:
    """
    检查训练标注几何详细信息

    Args:
        config: 配置字典

    Returns:
        几何检查结果
    """
    print("检查训练标注几何详情...")
    train_labels_path = config["data"]["train"]["labels"]
    gdf = load_vector(train_labels_path)
    result = check_vector_geometry(gdf)
    print(f"  总计: {result['total']}")
    print(f"  有效: {result['valid']}")
    print(f"  无效: {result['invalid']}")
    print(f"  空几何: {result['empty']}")
    return result


def run_full_check(config_path: str = "configs/config.yaml") -> DataCheckResult:
    """
    运行完整的数据检查

    Args:
        config_path: 配置文件路径

    Returns:
        DataCheckResult
    """
    print("=" * 60)
    print("数据完整性检查")
    print("=" * 60)

    config = load_config(config_path)
    all_issues = []

    # 检查训练数据
    print("\n[1/3] 检查训练数据...")
    train_image, train_labels, train_aoi, train_issues = check_train_data(config)
    all_issues.extend(train_issues)

    # 检查测试数据
    print("\n[2/3] 检查测试数据...")
    test_2023, test_2020, test_aoi, reg_check, test_issues = check_test_data(config)
    all_issues.extend(test_issues)

    # 检查标注几何
    print("\n[3/3] 检查标注几何...")
    geom_check = check_label_geometry(config)
    if geom_check["invalid"] > 0:
        all_issues.append(f"标注存在 {geom_check['invalid']} 个无效几何")

    # 汇总结果
    all_passed = len(all_issues) == 0

    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    if all_passed:
        print("✅ 所有检查通过")
    else:
        print(f"❌ 发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

    return DataCheckResult(
        train_image=train_image,
        train_labels=train_labels,
        train_aoi=train_aoi,
        test_image_2023=test_2023,
        test_image_2020=test_2020,
        test_aoi=test_aoi,
        registration_check=reg_check,
        geometry_check=geom_check,
        all_passed=all_passed,
        issues=all_issues,
    )


if __name__ == "__main__":
    result = run_full_check()
