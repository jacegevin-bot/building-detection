# CLAUDE.md - AI 工作上下文

## 项目简介

本项目是遥感综合实习第三个实战任务：**多时相影像的少样本建筑物提取与变化检测**。

## 项目目标

利用少量标注样本（训练区 999 个建筑物图斑），调用 SAM2 视觉大模型，完成：
1. 2023 年测试区建筑物提取
2. 2020-2023 年两期影像变化检测
3. 变化图斑清洗与矢量化输出

## 当前开发阶段

**核心流程已完成** — 待技术报告

## 当前项目状态

- [x] 数据集已准备（Dataset/ 目录）
- [x] 实习指导书已获取（Practical3_Guideline.pdf）
- [x] 项目文档初始化（README.md, CLAUDE.md, TODO.md）
- [x] 源代码目录结构创建
- [x] 配置文件创建（configs/config.yaml）
- [x] 技术决策已确定
- [x] 数据预处理代码（数据检查已通过）
- [x] 模型加载代码（SAM2 封装完成）
- [x] 建筑物提取代码（提取器已实现）
- [x] 变化检测代码（检测器已实现）
- [x] SAM2.1 权重下载（configs/sam2.1_hiera_base_plus.pt, 309MB）
- [x] 环境配置（conda rs3, Python 3.11, PyTorch 2.6+CUDA 12.4）
- [x] 任务 1：训练区伪标签 + 测试区 2023 建筑物提取
- [x] 任务 2：测试区 2020 建筑物提取 + 变化检测
- [x] 任务 3：形态学清洗 + 矢量化 → building_change.shp
- [ ] 技术报告（PDF）

## 工作流程（已实现）

```
训练标注 (999) ──Box/Point提示──→ SAM2 ──→ 训练影像 ──→ 伪标签 pseudo_label.tif
                                                                  │
        ┌─────────────────────────────────────────────────────────┘
        │
        ├──→ SAM2AutoMaskGenerator ──→ 测试影像 2023 ──→ building_2023.tif
        │    （分块处理 2048×2048, 20块）
        │
        └──→ SAM2AutoMaskGenerator ──→ 测试影像 2020 ──→ building_2020.tif

building_2023 XOR building_2020 ──→ 变化检测 ──→ 形态学清洗 ──→ 矢量化
                                                              → building_change.shp
```

**关键设计决策：** 训练区与测试区空间不重叠，因此测试区采用 SAM2AutomaticMaskGenerator 自动分割 + 训练区面积统计过滤的方案。

## 技术栈

### 环境
- **Conda 环境：** rs3（`~/miniconda3/envs/rs3/`）
- **Python：** 3.11
- **GPU：** RTX 4050 6GB（WSL2）

### 遥感数据处理
- rasterio - GeoTIFF 读写
- geopandas - 矢量数据处理
- shapely - 几何操作
- pyproj - 坐标转换

### 深度学习 / 视觉模型
- PyTorch 2.6 + CUDA 12.4
- SAM2.1 (hiera base plus) - 主流程视觉分割模型

### 图像处理
- scikit-image - 形态学操作
- scipy - 连通域分析
- numpy - 数组操作

## 技术决策（已确认）

### 模型选择
- **主流程：** SAM2.1 hiera base plus（config: `configs/sam2.1/sam2.1_hiera_b+`）

### 提示策略
- **训练区：** Box + 正负 Point 混合提示（从标注生成）
- **测试区：** SAM2AutomaticMaskGenerator 自动分割 + 面积过滤

### 分块策略
- **tile_size:** 2048×2048
- **overlap:** 256 像素
- **原因：** 测试影像 8454×5422 超出 6GB GPU 显存

### 形态学参数
- **min_area_m2:** 30.0 m²
- **min_pixels:** 50
- **struct_element_size:** 5

## 数据集信息

### 训练集
- **影像：** 2023 年 RGB，3 波段 Byte，2521×2944 像素
- **标注：** 999 个建筑物图斑
- **范围：** [114.4206, 30.4562, 114.4417, 30.4774]
- **CRS：** EPSG:4326

### 测试集
- **2023 影像：** 8454×5422 像素
- **2020 影像：** 8454×5422 像素（与 2023 同范围）
- **范围：** [114.4418, 30.4489, 114.5124, 30.4879]
- **标签：** 不提供，仅用于最终评估
- **训练区与测试区：** 不重叠（间隔约 0.00018°）

## 输出结果

| 文件 | 说明 |
|------|------|
| `results/masks/pseudo_label.tif` | 训练区伪标签（999 建筑物） |
| `results/masks/building_2023.tif` | 测试区 2023 建筑物掩膜 |
| `results/masks/building_2020.tif` | 测试区 2020 建筑物掩膜 |
| `results/masks/change_added_raw.tif` | 新增掩膜 |
| `results/masks/change_removed_raw.tif` | 拆除掩膜 |
| `results/masks/change_xor_raw.tif` | XOR 变化掩膜 |
| `results/vectors/building_change.shp` | 最终变化矢量（2047 图斑） |

## 运行命令

```bash
# 激活环境
export PATH="$HOME/miniconda3/envs/rs3/bin:$HOME/miniconda3/bin:$PATH"
export PYTHONPATH="<项目根目录>:$PYTHONPATH"

# 运行全部任务
python src/main.py --task 0

# 分步运行
python src/main.py --task 1                # 训练区伪标签 + 测试区 2023
python src/main.py --task 1 --skip-pseudo  # 跳过伪标签（如已存在）
python src/main.py --task 2                # 测试区 2020 + 变化检测
python src/main.py --task 3                # 形态学 + 矢量化
```

## 已知问题

- SAM2 `_C` 模块导入警告（不影响结果，缺少 C++ 后处理加速）
- D: 盘在 WSL2 中访问不稳定，项目文件在 `~/rs3_project/` 工作

## 注意事项

1. 测试区建筑物标签**绝对不得**用于任何形式的训练或调参
2. 2020 年影像仅用于变化检测阶段
3. 两期影像和掩膜需保持在同一空间网格
4. 最终 Shapefile 需包含完整属性字段
5. 形态学参数需与影像分辨率匹配
