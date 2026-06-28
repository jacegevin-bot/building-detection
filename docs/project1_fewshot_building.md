# Project 1: Few-Shot Building Extraction & Change Detection

## Overview

Multi-temporal image based few-shot building extraction and change detection using SAM2 vision foundation model. This project demonstrates how to leverage few-shot learning with foundation models for remote sensing applications.

## Objectives

1. **Task 1:** Few-shot building extraction on 2023 test area
2. **Task 2:** Two-phase change detection (2020-2023)
3. **Task 3:** Morphological cleaning and Shapefile output

## Methodology

### Workflow

```
Training Annotations (999) ──Box/Point Prompts──→ SAM2 ──→ Training Image ──→ Pseudo Labels
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        │
        ├──→ SAM2AutoMaskGenerator ──→ Test Image 2023 ──→ building_2023.tif
        │    (Tiled: 2048×2048, 20 tiles)
        │
        └──→ SAM2AutoMaskGenerator ──→ Test Image 2020 ──→ building_2020.tif

building_2023 XOR building_2020 ──→ Change Detection ──→ Morphological Cleaning ──→ Vectorization
                                                                                    → building_change.shp
```

### Key Design Decisions

1. **Spatial Separation:** Training and test areas do not overlap (0.00018° gap)
2. **Automatic Segmentation:** Test area uses SAM2AutomaticMaskGenerator + area filtering
3. **Tiled Processing:** 8454×5422 images exceed 6GB GPU memory → 2048×2048 tiles with 256px overlap

## Dataset

| Item | Path | Description |
|------|------|-------------|
| Training Image | `data/train/train_image_2023_independent_roof.tif` | 2023 RGB, 2521×2944 |
| Training Labels | `data/train/buildings_train_independent_roof.shp` | 999 building polygons |
| Test Image 2023 | `data/test/test_image_2023_independent_roof.tif` | 8454×5422 |
| Test Image 2020 | `data/test/test_image_2020_independent_roof.tif` | 8454×5422 |

**Note:** Training and test areas are spatially separated. Test labels are not provided.

## Results

### Quantitative Results

| Metric | Value |
|--------|-------|
| Training area pseudo-labels | 999 buildings |
| Test 2023 non-zero pixels | 15,457,019 |
| Test 2020 non-zero pixels | 19,616,650 |
| New building pixels | 4,409,664 |
| Demolished building pixels | 8,569,295 |
| Total change patches | 2,047 |
| New patches | 892 |
| Demolished patches | 1,073 |

### Output Files

| File | Description |
|------|-------------|
| `results/masks/pseudo_label.tif` | Training area pseudo-labels |
| `results/masks/building_2023.tif` | Test area 2023 buildings |
| `results/masks/building_2020.tif` | Test area 2020 buildings |
| `results/masks/change_added_raw.tif` | New buildings mask |
| `results/masks/change_removed_raw.tif` | Demolished buildings mask |
| `results/vectors/building_change.shp` | Final change vector (2047 patches) |

### Change Vector Attributes

| Field | Type | Description |
|-------|------|-------------|
| change_id | Integer | Change patch ID |
| chg_type | String | Change type (added/removed/changed) |
| area_m2 | Float | Area in square meters |

## Technical Details

### Model Configuration

- **Model:** SAM2.1 hiera base plus
- **Config:** `configs/sam2.1/sam2.1_hiera_b+`
- **Weights:** `configs/sam2.1_hiera_base_plus.pt` (309MB)

### Prompt Strategy

- **Training area:** Box + 5 positive points + 3 negative points
- **Test area:** Automatic mask generation + area filtering

### Morphological Parameters

- **min_area_m2:** 30.0 m²
- **min_pixels:** 50
- **struct_element_size:** 5

## Running the Pipeline

```bash
# Set environment
export PATH="$HOME/miniconda3/envs/rs3/bin:$HOME/miniconda3/bin:$PATH"
export PYTHONPATH="<project_root>:$PYTHONPATH"

# Run all tasks
python src/main.py --task 0

# Run specific tasks
python src/main.py --task 1                # Pseudo-label + test 2023
python src/main.py --task 1 --skip-pseudo  # Skip existing pseudo-labels
python src/main.py --task 2                # Test 2020 + change detection
python src/main.py --task 3                # Morphological cleaning + vectorization
```

## Known Issues

- SAM2 `_C` module import warning (does not affect results)
- CUDA OOM for full-size images → solved with tiled processing

## Future Work

- [ ] RS-SAM ablation study (IoU, miss rate)
- [ ] Pure Box vs Pure Mask prompt comparison
- [ ] LoRA fine-tuning experiment
- [ ] LoFTR registration comparison
- [ ] Visualization and final report

## References

- SAM2: [Segment Anything Model 2](https://github.com/facebookresearch/sam2)
- RS-SAM: [Remote Sensing SAM](https://github.com/...)

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-06-28 | Environment setup, code development |
| 2026-06-28 | Core pipeline completed |
| TBD | Ablation experiments |
| TBD | Final report |
