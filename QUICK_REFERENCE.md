# OCICalc Quick Reference

## Installation
```bash
pip install numpy pandas matplotlib czifile cellpose opencv-python PyYAML scikit-image
```

## Usage

### Process Single File
```bash
python ocicalc.py /path/to/image.czi
```

### Process Folder
```bash
python ocicalc.py -f /path/to/folder/
```

### Get Help
```bash
python ocicalc.py --help
```

## Outputs (per .czi file)

| File | Content |
|------|---------|
| `<name>_mask1.png` | Cell segmentation mask (channel 1) |
| `<name>_mask2.png` | Cell segmentation mask (channel 2) |
| `<name>_cell_wise.csv` | Per-cell measurements |

## CSV Columns

- `cell_id` - Cell identifier
- `channel1_mean` - Mean intensity (ch1)
- `channel2_mean` - Mean intensity (ch2)
- `channel1_max` - Max intensity (ch1)
- `channel2_max` - Max intensity (ch2)
- `area_channel1_mask` - Mask area (pixels)
- `area_channel2_mask` - Mask area (pixels)
- `oci_ratio` - ch2_mean / ch1_max

## Requirements

- `helper.py` in Python path
- `parameters.yaml` at `/mnt/project/parameters.yaml`
- CUDA for GPU acceleration (optional)

## Common Issues

**Import errors**: Install missing packages with pip  
**helper.py not found**: Add project dir to PYTHONPATH  
**GPU errors**: Change `gpu=True` to `gpu=False` in script  
**Parameters not found**: Update path in `load_parameters()`

## Key Settings

- **DPI**: 150 (line 53)
- **Colormap**: nipy_spectral (line 54)
- **Folder mode**: Non-recursive
- **Cellpose model**: Defined in parameters.yaml

## File Structure
```
ocicalc.py (345 lines)
├── CZIProcessor class
│   ├── __init__
│   ├── process_czi
│   ├── calculate_measurements
│   ├── save_mask_images
│   └── save_measurements
└── Functions
    ├── main
    ├── load_parameters
    ├── process_file
    ├── find_czi_files
    └── validate_input
```
