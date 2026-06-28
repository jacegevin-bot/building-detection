# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-28

### Added
- Initial project structure
- SAM2 wrapper for building extraction
- Prompt generator (Box + Point hybrid)
- Building extractor with tiled processing
- Change detector (XOR-based)
- Morphological cleaning pipeline
- Vectorization to Shapefile
- Data validation scripts

### Completed
- Training area pseudo-label generation (999 buildings)
- Test area 2023 building extraction (15.5M pixels)
- Test area 2020 building extraction (19.6M pixels)
- Change detection: 892 new + 1,073 demolished buildings
- Final output: 2,047 change patches in Shapefile

### Technical Details
- Model: SAM2.1 hiera base plus
- Tiled processing: 2048×2048 with 256px overlap
- Morphological params: min_area=30m², min_pixels=50

## [Unreleased]

### Planned
- RS-SAM ablation study
- Pure Box vs Pure Mask prompt comparison
- LoRA fine-tuning experiment
- LoFTR registration comparison
- Visualization and final report
