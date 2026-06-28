#!/usr/bin/env python3
"""
遥感综合实习实战三 - 多时相影像的少样本建筑物提取与变化检测
主入口脚本

工作流程（根据 Practical3_Guideline.pdf）：
  Step 1: 训练区建筑物提取 → 伪标签
    - 训练标注 → Box/Point 提示 → SAM2 → 训练影像 → pseudo_label.tif
  Step 2: 测试区 2023 建筑物提取
    - 伪标签 → Mask 提示 → SAM2 → 测试影像 2023 → building_2023.tif
  Step 3: 测试区 2020 建筑物提取
    - 伪标签 → Mask 提示 → SAM2 → 测试影像 2020 → building_2020.tif
  Step 4: 变化检测
    - XOR(building_2020, building_2023) → change map
  Step 5: 形态学清洗 + 矢量化
    - 开闭运算、孔洞填充、小斑块过滤 → building_change.shp
"""

import argparse
import time
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_sam2_model(config: dict):
    """创建并加载 SAM2 模型"""
    from src.model.sam2_wrapper import SAM2Config, SAM2Wrapper

    model_config = SAM2Config(
        model_type=config["model"]["primary"]["variant"],
        checkpoint=config["model"]["primary"]["checkpoint"],
        device=config["model"]["primary"]["device"],
    )
    model = SAM2Wrapper(model_config)
    model.load_model()
    return model


def run_step1_pseudo_labels(config: dict, model) -> str:
    """
    Step 1: 训练区建筑物提取 → 伪标签

    用训练区标注（Box + Point 提示）在训练影像上运行 SAM2，
    得到训练区的建筑物伪标签掩膜。

    Returns:
        伪标签掩膜路径
    """
    print("\n" + "=" * 60)
    print("Step 1: 训练区建筑物提取 → 伪标签")
    print("=" * 60)

    import numpy as np
    import rasterio
    import geopandas as gpd

    from src.extraction.prompt_generator import generate_prompts
    from src.extraction.building_extractor import ExtractionConfig

    train_image_path = config["data"]["train"]["image"]
    train_labels_path = config["data"]["train"]["labels"]
    output_path = config["output"]["masks"].get("pseudo_label", "results/masks/pseudo_label.tif")

    # 读取训练影像
    print(f"读取训练影像: {train_image_path}")
    with rasterio.open(train_image_path) as src:
        image = src.read().transpose(1, 2, 0)  # (H, W, C)
        transform = src.transform
        profile = src.profile.copy()
        print(f"  影像尺寸: {src.width} x {src.height}")

    # 读取训练标注
    labels_gdf = gpd.read_file(train_labels_path)
    print(f"训练标注数量: {len(labels_gdf)}")

    # 生成提示（Box + Point 混合）
    strategy = config["prompt"]["primary"]["strategy"]
    print(f"生成提示（策略: {strategy}）...")
    prompts = generate_prompts(
        gdf=labels_gdf,
        transform=transform,
        image_shape=(profile["height"], profile["width"]),
        strategy=strategy,
        num_positive=config["prompt"]["primary"]["num_positive_points"],
        num_negative=config["prompt"]["primary"]["num_negative_points"],
    )
    print(f"共生成 {len(prompts.prompts)} 个提示")

    # 逐个建筑物推理
    print("开始提取建筑物...")
    h, w = profile["height"], profile["width"]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    success_count = 0

    extract_config = ExtractionConfig(
        prompt_strategy=strategy,
        confidence_threshold=0.5,
    )

    for i, prompt in enumerate(prompts.prompts):
        try:
            model.set_image(image)

            if prompt.bbox is not None and len(prompt.positive_points) > 0:
                result = model.predict_hybrid(
                    box=prompt.bbox,
                    points=prompt.positive_points,
                    labels=np.ones(len(prompt.positive_points)),
                    multimask_output=False,
                )
            elif prompt.bbox is not None:
                result = model.predict_with_box(
                    box=prompt.bbox,
                    multimask_output=False,
                )
            else:
                continue

            best_idx = np.argmax(result.scores)
            mask = result.masks[best_idx]
            score = result.scores[best_idx]

            if score >= extract_config.confidence_threshold:
                final_mask = np.maximum(final_mask, mask.astype(np.uint8))
                confidence_map = np.maximum(confidence_map, mask * score)
                success_count += 1

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(prompts.prompts)} 个建筑物")

        except Exception as e:
            print(f"  ⚠️ 建筑物 {i} 提取失败: {e}")
            continue

    print(f"✅ 伪标签生成完成: {success_count}/{len(prompts.prompts)} 个建筑物")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_mask, 1)
    print(f"✅ 伪标签已保存: {output_path}")

    return str(output_path)


def run_step2_test_extraction(config: dict, model, pseudo_label_path: str,
                               image_key: str = "image_2023",
                               output_key: str = "building_2023") -> str:
    """
    Step 2/3: 在测试影像上提取建筑物

    策略：使用 SAM2 自动分割（AutomaticMaskGenerator）在测试影像上
    生成所有候选掩膜，然后根据训练区建筑物的面积统计特征进行过滤。

    Args:
        config: 配置
        model: SAM2 模型（SAM2Wrapper）
        pseudo_label_path: 伪标签路径（用于统计训练区建筑物特征）
        image_key: 测试影像键名 ("image_2023" 或 "image_2020")
        output_key: 输出掩膜键名 ("building_2023" 或 "building_2020")

    Returns:
        输出掩膜路径
    """
    import numpy as np
    import rasterio
    from scipy import ndimage
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image_path = config["data"]["test"][image_key]
    output_path = config["output"]["masks"][output_key]

    # --- 从伪标签统计训练区建筑物特征 ---
    print(f"\n分析训练区建筑物特征: {pseudo_label_path}")
    with rasterio.open(pseudo_label_path) as src:
        pseudo_mask = src.read(1)
    labeled_train, num_train = ndimage.label(pseudo_mask)
    areas_train = []
    for i in range(1, num_train + 1):
        areas_train.append(np.count_nonzero(labeled_train == i))
    areas_train = np.array(areas_train)
    # 过滤掉极小的噪声区域
    valid_areas = areas_train[areas_train >= 10]
    min_building_pixels = max(30, int(np.percentile(valid_areas, 5)))  # 第5百分位
    print(f"  训练区建筑物数: {num_train}")
    print(f"  有效建筑物面积范围: {valid_areas.min()} ~ {valid_areas.max()} 像素")
    print(f"  过滤阈值 (5th percentile): {min_building_pixels} 像素")

    # --- 读取测试影像 ---
    print(f"\n读取测试影像: {image_path}")
    with rasterio.open(image_path) as src:
        image = src.read().transpose(1, 2, 0)  # (H, W, C)
        transform = src.transform
        profile = src.profile.copy()
        h, w = src.height, src.width
        print(f"  影像尺寸: {w} x {h}")

    # --- 分块 SAM2 自动分割 ---
    # 测试影像太大（8454×5422），需要分块处理以避免 CUDA OOM
    tile_size = 2048
    overlap = 256
    print(f"\n分块 SAM2 自动分割（tile={tile_size}, overlap={overlap}）...")

    mask_generator = SAM2AutomaticMaskGenerator(
        model=model.model,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=min_building_pixels,
        box_nms_thresh=0.7,
    )

    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 计算分块
    tiles = []
    for y0 in range(0, h, tile_size - overlap):
        for x0 in range(0, w, tile_size - overlap):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tiles.append((y0, y1, x0, x1))

    print(f"  总分块数: {len(tiles)}")

    for tile_idx, (y0, y1, x0, x1) in enumerate(tiles):
        tile_img = image[y0:y1, x0:x1].copy()
        tile_h, tile_w = tile_img.shape[:2]

        try:
            masks_data = mask_generator.generate(tile_img)
            tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)

            kept = 0
            for item in masks_data:
                if item["area"] < min_building_pixels:
                    continue
                if item["predicted_iou"] < 0.7 or item["stability_score"] < 0.85:
                    continue
                tile_mask = np.maximum(tile_mask, item["segmentation"].astype(np.uint8))
                kept += 1

            # 合并到全局掩膜（中间区域去掉 overlap 边界）
            # 计算有效区域（去掉 overlap 的一半）
            margin = overlap // 2 if tile_idx > 0 else 0
            ey0 = y0 + (margin if y0 > 0 else 0)
            ey1 = y1 - (margin if y1 < h else 0)
            ex0 = x0 + (margin if x0 > 0 else 0)
            ex1 = x1 - (margin if x1 < w else 0)

            # 对应 tile 内的偏移
            ty0 = ey0 - y0
            ty1 = ey1 - y0
            tx0 = ex0 - x0
            tx1 = ex1 - x0

            final_mask[ey0:ey1, ex0:ex1] = np.maximum(
                final_mask[ey0:ey1, ex0:ex1],
                tile_mask[ty0:ty1, tx0:tx1],
            )

            print(f"  [{tile_idx+1}/{len(tiles)}] ({y0},{x0})-{y1},{x1}: {kept} 个掩膜")

        except Exception as e:
            print(f"  [{tile_idx+1}/{len(tiles)}] ⚠️ 失败: {e}")
            continue

        finally:
            # 释放显存
            del tile_img
            import gc
            gc.collect()

    kept_count = np.count_nonzero(final_mask)
    print(f"✅ 测试区提取完成: 非零像元 {kept_count}")

    # --- 保存 ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_mask, 1)
    print(f"✅ 掩膜已保存: {output_path}")

    return str(output_path)


def run_step4_change_detection(config: dict, mask_2023_path: str, mask_2020_path: str):
    """
    Step 4: 变化检测（XOR）

    Returns:
        (change_added_path, change_removed_path, change_xor_path)
    """
    import numpy as np
    import rasterio

    print("\n" + "=" * 60)
    print("Step 4: 变化检测")
    print("=" * 60)

    # 读取两期掩膜
    with rasterio.open(mask_2023_path) as src:
        mask_2023 = src.read(1)
        profile = src.profile.copy()

    with rasterio.open(mask_2020_path) as src:
        mask_2020 = src.read(1)

    # 二值化
    mask_2023 = (mask_2023 > 0).astype(np.uint8)
    mask_2020 = (mask_2020 > 0).astype(np.uint8)

    # 变化检测
    added = ((mask_2023 == 1) & (mask_2020 == 0)).astype(np.uint8)    # 新增
    removed = ((mask_2020 == 1) & (mask_2023 == 0)).astype(np.uint8)  # 拆除
    xor = ((mask_2023 != mask_2020)).astype(np.uint8)                  # 总变化

    print(f"新增像元: {np.count_nonzero(added)}")
    print(f"拆除像元: {np.count_nonzero(removed)}")
    print(f"总变化像元: {np.count_nonzero(xor)}")

    # 保存
    output_dir = Path("results/masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile.update(count=1, dtype="uint8", nodata=0)

    paths = {}
    for name, data in [("change_added_raw", added), ("change_removed_raw", removed), ("change_xor_raw", xor)]:
        path = output_dir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        paths[name] = str(path)
        print(f"✅ 已保存: {path}")

    return paths["change_added_raw"], paths["change_removed_raw"], paths["change_xor_raw"]


def run_step5_morphology_and_vectorize(config: dict, change_added_path: str, change_removed_path: str):
    """
    Step 5: 形态学清洗 + 矢量化 → building_change.shp
    """
    import numpy as np
    import rasterio
    import geopandas as gpd
    from shapely.geometry import shape
    from rasterio.features import shapes
    from scipy import ndimage

    print("\n" + "=" * 60)
    print("Step 5: 形态学清洗 + 矢量化")
    print("=" * 60)

    morpho_config = config["morphology"]
    min_area_m2 = morpho_config["min_area_m2"]
    min_pixels = morpho_config["min_pixels"]
    struct_size = morpho_config["struct_element_size"]

    # 读取变化掩膜
    with rasterio.open(change_added_path) as src:
        added_raw = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    with rasterio.open(change_removed_path) as src:
        removed_raw = src.read(1)

    def clean_mask(mask, label=""):
        """形态学清洗"""
        from skimage.morphology import binary_opening, binary_closing

        # 开运算去噪
        struct = np.ones((struct_size, struct_size))
        cleaned = binary_opening(mask, struct)
        # 闭运算填补缝隙
        cleaned = binary_closing(cleaned, struct)

        # 孔洞填充
        if morpho_config["fill_holes"]:
            cleaned = ndimage.binary_fill_holes(cleaned)

        # 小斑块过滤
        labeled, num = ndimage.label(cleaned)
        for region_id in range(1, num + 1):
            region = (labeled == region_id)
            if np.count_nonzero(region) < min_pixels:
                cleaned[region] = 0

        print(f"  {label}: 清洗前 {np.count_nonzero(mask)} 像元 → 清洗后 {np.count_nonzero(cleaned)} 像元")
        return cleaned.astype(np.uint8)

    print("形态学清洗...")
    added_clean = clean_mask(added_raw, "新增")
    removed_clean = clean_mask(removed_raw, "拆除")

    # 合并为总变化掩膜
    total_change = np.maximum(added_clean, removed_clean)

    # 矢量化
    print("矢量化...")
    results = []
    for geom_dict, value in shapes(total_change, transform=transform):
        if value == 0:
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue

        # 判断是新增还是拆除
        # 用质心判断
        centroid = geom.centroid
        col, row = ~transform * (centroid.x, centroid.y)
        col, row = int(col), int(row)

        if 0 <= row < added_clean.shape[0] and 0 <= col < added_clean.shape[1]:
            if added_clean[row, col] == 1:
                chg_type = "added"
            elif removed_clean[row, col] == 1:
                chg_type = "removed"
            else:
                chg_type = "changed"
        else:
            chg_type = "changed"

        # 计算面积（度→平方米，使用 WGS84 近似）
        # 在纬度30°附近，1度 ≈ 96km 经度, 111km 纬度
        area_deg2 = geom.area
        lat_rad = np.radians(centroid.y)
        m_per_deg_lon = 111320 * np.cos(lat_rad)
        m_per_deg_lat = 110540
        area_m2 = area_deg2 * m_per_deg_lon * m_per_deg_lat

        results.append({
            "geometry": geom,
            "chg_type": chg_type,
            "area_m2": area_m2,
        })

    if not results:
        print("⚠️ 未检测到变化图斑")
        return None

    gdf = gpd.GeoDataFrame(results, crs=crs)

    # 过滤面积太小的图斑
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()

    # 添加属性
    gdf["change_id"] = range(1, len(gdf) + 1)
    gdf = gdf[["change_id", "chg_type", "area_m2", "geometry"]]

    # 保存
    output_path = Path(config["output"]["vectors"]["building_change"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, encoding="utf-8")

    print(f"✅ 矢量化完成")
    print(f"   总图斑数: {len(gdf)}")
    print(f"   新增: {len(gdf[gdf['chg_type'] == 'added'])}")
    print(f"   拆除: {len(gdf[gdf['chg_type'] == 'removed'])}")
    print(f"   输出: {output_path}")

    return str(output_path)


def run_task1(config: dict, skip_pseudo: bool = False) -> None:
    """任务 1：训练区建筑物提取 + 测试区 2023 建筑物提取"""
    model = create_sam2_model(config)

    pseudo_label_path = config["output"]["masks"].get("pseudo_label", "results/masks/pseudo_label.tif")

    # Step 1: 训练区 → 伪标签
    if skip_pseudo and Path(pseudo_label_path).exists():
        print(f"\n跳过 Step 1（伪标签已存在: {pseudo_label_path}）")
    else:
        pseudo_label_path = run_step1_pseudo_labels(config, model)

    # Step 2: 伪标签 → 测试区 2023
    run_step2_test_extraction(config, model, pseudo_label_path,
                               image_key="image_2023", output_key="building_2023")


def run_task2(config: dict) -> None:
    """任务 2：测试区 2020 建筑物提取 + 变化检测"""
    model = create_sam2_model(config)

    pseudo_label_path = config["output"]["masks"].get("pseudo_label", "results/masks/pseudo_label.tif")

    # Step 3: 伪标签 → 测试区 2020
    run_step2_test_extraction(config, model, pseudo_label_path,
                               image_key="image_2020", output_key="building_2020")

    # Step 4: 变化检测
    run_step4_change_detection(
        config,
        config["output"]["masks"]["building_2023"],
        config["output"]["masks"]["building_2020"],
    )


def run_task3(config: dict) -> None:
    """任务 3：形态学清洗 + 矢量化"""
    run_step5_morphology_and_vectorize(
        config,
        config["output"]["change"]["added_raw"],
        config["output"]["change"]["removed_raw"],
    )


def run_ablation(config: dict) -> None:
    """消融实验"""
    print("=" * 60)
    print("消融实验")
    print("=" * 60)
    raise NotImplementedError("消融实验尚未实现")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="遥感综合实习实战三 - 多时相影像的少样本建筑物提取与变化检测"
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 0],
        required=True,
        help="任务编号（1/2/3），0=运行全部",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="运行消融实验",
    )
    parser.add_argument(
        "--skip-pseudo",
        action="store_true",
        help="跳过伪标签生成（如果已存在）",
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print(f"配置已加载: {args.config}")

    t0 = time.time()

    # 运行任务
    if args.task == 0:
        run_task1(config, skip_pseudo=args.skip_pseudo)
        run_task2(config)
        run_task3(config)
    elif args.task == 1:
        run_task1(config, skip_pseudo=args.skip_pseudo)
    elif args.task == 2:
        run_task2(config)
    elif args.task == 3:
        run_task3(config)

    # 运行消融实验
    if args.ablation:
        run_ablation(config)

    elapsed = time.time() - t0
    print(f"\n✅ 所有任务完成 (耗时 {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
