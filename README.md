# Building Detection

Building extraction and detection using computer vision and deep learning.

## Overview

This repository documents my journey in building detection research and applications, combining remote sensing, computer vision, and deep learning techniques.

## Projects

### 1. Few-Shot Building Extraction & Change Detection

**Status:** Core pipeline completed ✅ | Ablation experiments & report pending

Multi-temporal image based few-shot building extraction and change detection using SAM2 vision foundation model.

**Key Results:**
- Training area: 999 buildings extracted via Box+Point prompts
- Test area: 2023 buildings (15.5M pixels), 2020 buildings (19.6M pixels)
- Change detection: 892 new buildings, 1,073 demolished
- Final output: 2,047 change patches in Shapefile

**Tech Stack:** SAM2.1, PyTorch, Rasterio, GeoPandas, scikit-image

[→ Full Documentation](docs/project1_fewshot_building.md)

## Repository Structure

```
building-detection/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── configs/                           # Configuration files
│   └── config.yaml                    # Main config
├── src/                               # Source code
│   ├── main.py                        # Main entry point
│   ├── model/                         # Model wrappers
│   │   ├── sam2_wrapper.py            # SAM2 wrapper
│   │   └── rssam_wrapper.py           # RS-SAM wrapper (ablation)
│   ├── extraction/                    # Building extraction
│   │   ├── prompt_generator.py        # Prompt generation
│   │   └── building_extractor.py      # Building extractor
│   ├── change_detection/              # Change detection
│   │   └── change_detector.py         # Change detector
│   ├── preprocessing/                 # Data preprocessing
│   │   └── data_checker.py            # Data validation
│   ├── postprocessing/                # Post-processing
│   ├── visualization/                 # Visualization tools
│   └── utils/                         # Utilities
│       ├── raster_utils.py            # Raster utilities
│       └── vector_utils.py            # Vector utilities
├── results/                           # Output results
│   ├── masks/                         # Building masks
│   ├── vectors/                       # Vector outputs
│   ├── figures/                       # Visualizations
│   └── metrics/                       # Evaluation metrics
├── experiments/                       # Ablation experiments
│   ├── ablation_01_rssam/             # RS-SAM comparison
│   ├── ablation_02_prompt_strategy/   # Prompt strategy comparison
│   └── ablation_03_lora/              # LoRA fine-tuning
├── scripts/                           # Utility scripts
│   ├── check_data.py                  # Data validation
│   └── download_models.py             # Model download
├── docs/                              # Documentation
└── data/                              # Dataset (not tracked)
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/jacegevin-bot/building-detection.git
cd building-detection

# Create environment
conda create -n rs3 python=3.11
conda activate rs3

# Install dependencies
pip install -r requirements.txt

# Download SAM2 weights
python scripts/download_models.py

# Run pipeline
python src/main.py --task 0
```

## Progress Log

| Date | Progress |
|------|----------|
| 2026-06-28 | Core pipeline completed: pseudo-label → building extraction → change detection → vectorization |

## Tech Stack

- **Deep Learning:** PyTorch 2.6 + CUDA 12.4, SAM2.1
- **Remote Sensing:** Rasterio, GeoPandas, Shapely, PyProj
- **Image Processing:** scikit-image, SciPy, OpenCV
- **Visualization:** Matplotlib, Seaborn

## License

MIT License

## Contact

GitHub: [@jacegevin-bot](https://github.com/jacegevin-bot)
