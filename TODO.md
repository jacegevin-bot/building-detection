# TODO.md - 任务清单与日志

---

## ✅ 阶段 0-2：环境与代码准备 (2026-06-28)

- [x] 项目初始化：目录结构、配置文件、.gitignore
- [x] 数据集检查与元信息读取
- [x] 实习指导书阅读与理解
- [x] 项目文档创建（README.md, CLAUDE.md, TODO.md）
- [x] 源代码模块编写
  - `src/utils/raster_utils.py` - 栅格数据工具
  - `src/utils/vector_utils.py` - 矢量数据工具
  - `src/preprocessing/data_checker.py` - 数据检查模块
  - `src/extraction/prompt_generator.py` - 提示生成模块
  - `src/model/sam2_wrapper.py` - SAM2 模型封装
  - `src/extraction/building_extractor.py` - 建筑物提取器
  - `src/change_detection/change_detector.py` - 变化检测器
- [x] 数据完整性验证通过

---

## ✅ 阶段 3-6：核心流程 (2026-06-28)

### 环境配置
- [x] 下载 SAM2.1-base-plus 权重 → `configs/sam2.1_hiera_base_plus.pt` (309MB)
- [x] 安装 Miniconda（WSL2）+ 创建 rs3 环境（Python 3.11）
- [x] 安装 PyTorch 2.6 + CUDA 12.4
- [x] 安装 SAM2 库（`pip install --no-build-isolation`）
- [x] 安装全部依赖（rasterio, geopandas, scikit-image 等）

### 工作流程修正
- [x] 发现训练区与测试区不重叠（经度 114.42~114.44 vs 114.44~114.51）
- [x] 修正方案：测试区改用 SAM2AutomaticMaskGenerator 自动分割 + 训练区面积统计过滤

### 阶段 3：训练区伪标签生成
- [x] Box + Point 混合提示 → SAM2 → 训练影像
- [x] 输出 `pseudo_label.tif`（999/999 建筑物成功提取）

### 阶段 4：测试区建筑物提取
- [x] 分块自动分割（tile=2048, overlap=256, 共 20 块）
- [x] `building_2023.tif`（15,457,019 非零像元）
- [x] `building_2020.tif`（19,616,650 非零像元）
- [x] 解决 CUDA OOM 问题（6GB 显存 vs 8454×5422 影像）

### 阶段 5：变化检测
- [x] XOR 运算
- [x] `change_added_raw.tif`（4,409,664 新增像元）
- [x] `change_removed_raw.tif`（8,569,295 拆除像元）
- [x] `change_xor_raw.tif`（12,978,959 总变化像元）

### 阶段 6：形态学清洗 + 矢量化
- [x] 开闭运算、孔洞填充、小斑块过滤
- [x] 栅格转矢量、面积计算
- [x] `building_change.shp`（2,047 个图斑）
  - 新增: 892 个
  - 拆除: 1,073 个
  - 变化: 82 个

---

## 📋 待完成

### 可视化图件
- [x] 新增/拆除分布专题图 (2026-06-29)
- [x] 典型区域叠加图 (2026-06-29)
- [x] 变化与伪变化对比图 (2026-06-29)

### 技术报告
- [x] 编写实验报告（PDF）(2026-06-29)

---

## 📝 运行日志

### 2026-06-28 晚

**Step 1 - 训练区伪标签生成：**
```
配置: SAM2.1 hiera base_plus, CUDA
提示策略: box_point_hybrid (5正点+3负点)
结果: 999/999 建筑物提取成功
输出: results/masks/pseudo_label.tif
耗时: ~8 分钟
```

**Step 2 - 测试区 2023 建筑物提取：**
```
方案: SAM2AutomaticMaskGenerator 分块处理
分块: 2048×2048, overlap=256, 共 20 块
过滤: min_mask_region_area=30px, pred_iou_thresh=0.7, stability_score_thresh=0.85
结果: 15,457,019 非零像元
输出: results/masks/building_2023.tif
耗时: ~57 秒
```

**Step 3 - 测试区 2020 建筑物提取：**
```
方案: 同 Step 2
结果: 19,616,650 非零像元
输出: results/masks/building_2020.tif
耗时: ~57 秒
```

**Step 4 - 变化检测：**
```
新增像元: 4,409,664
拆除像元: 8,569,295
总变化像元: 12,978,959
输出: change_added_raw.tif, change_removed_raw.tif, change_xor_raw.tif
耗时: ~1 秒
```

**Step 5 - 形态学清洗 + 矢量化：**
```
参数: min_area_m2=30, min_pixels=50, struct_size=5
新增清洗: 4,409,664 → 4,463,208 像元
拆除清洗: 8,569,295 → 9,154,669 像元
矢量化: 2,047 个图斑 (892 新增 + 1,073 拆除 + 82 变化)
输出: results/vectors/building_change.shp
耗时: ~81 秒
```

**总耗时: ~12 分钟**

### 2026-06-29

**可视化图件生成：**
```
脚本: scripts/generate_figures.py
输出:
  - results/figures/change_distribution_map.png (10.6 MB)
  - results/figures/typical_area_overlay.png (7.6 MB)
  - results/figures/change_comparison.png (2.0 MB)
耗时: ~30 秒
```

**GitHub 仓库构建：**
```
仓库: https://github.com/jacegevin-bot/building-detection
内容: 完整项目框架、源代码、文档、可视化图件
提交: feat: Complete project structure for few-shot building extraction
```

---

## 🐛 已解决问题

1. **Python 3.14 不兼容 PyTorch** → 安装 Miniconda + rs3 环境（Python 3.11）
2. **SAM2 配置名错误** → 需要 `configs/sam2.1/sam2.1_hiera_b+` 前缀
3. **训练区与测试区不重叠** → 改用自动分割 + 统计过滤方案
4. **CUDA OOM（6GB vs 8454×5422）** → 分块处理（2048×2048）
5. **SAM2 mask 输入维度** → 需要 (1, H, W) 格式
6. **D 盘 WSL 访问不稳定** → 项目复制到 ~/rs3_project/ 工作

---

## 🔮 未来改进（可选）

- [ ] 尝试不同 SAM 变体对比效果
- [ ] 优化形态学参数
