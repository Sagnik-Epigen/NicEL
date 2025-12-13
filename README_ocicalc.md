# OCICalc - Optical Cell Index Calculator

A terminal-runnable Python script for processing `.czi` microscopy images to extract per-cell measurements and generate mask visualizations.

## Overview

This script reproduces the analysis workflow from `OCICalc.ipynb` in a production-ready, command-line interface. It processes CZI microscopy files using Cellpose for cell segmentation and calculates per-cell intensity measurements.

## Features

- ✅ **Terminal-based**: Run from command line with arguments
- ✅ **Single file or batch processing**: Process one file or an entire folder
- ✅ **Robust error handling**: Continues processing even if individual files fail
- ✅ **Detailed logging**: Progress tracking and error reporting
- ✅ **Standardized outputs**: PNG masks and CSV measurements for each input

## Installation

### Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib czifile cellpose opencv-python PyYAML scikit-image
```

### Project Files

Ensure the following project files are accessible:
- `helper.py` - Must be in Python path or same directory
- `parameters.yaml` - Must be at `/mnt/project/parameters.yaml` (or modify path in script)

## Usage

### Command Line Interface

**Process a single .czi file:**
```bash
python ocicalc.py /path/to/image.czi
```

**Process all .czi files in a folder (non-recursive):**
```bash
python ocicalc.py -f /path/to/folder
```

**View help:**
```bash
python ocicalc.py --help
```

### Arguments

- `path` (positional): Path to .czi file or folder
- `-f`, `--folder` (optional): Treat path as folder containing .czi files

## Outputs

For each processed `.czi` file, the script generates three output files:

### 1. Mask Images (PNG)
- **`<filename>_mask1.png`**: Visualization of cell segmentation masks (channel 1)
- **`<filename>_mask2.png`**: Visualization of cell segmentation masks (channel 2)
- Rendered using `nipy_spectral` colormap at 150 DPI

### 2. Cell Measurements (CSV)
- **`<filename>_cell_wise.csv`**: Per-cell quantitative measurements

#### CSV Columns:
| Column | Description |
|--------|-------------|
| `cell_id` | Unique cell identifier (1-indexed) |
| `channel1_mean` | Mean intensity in channel 1 |
| `channel2_mean` | Mean intensity in channel 2 |
| `channel1_max` | Maximum intensity in channel 1 |
| `channel2_max` | Maximum intensity in channel 2 |
| `area_channel1_mask` | Number of pixels in cell mask (channel 1) |
| `area_channel2_mask` | Number of pixels in cell mask (channel 2) |
| `oci_ratio` | Optical cell index ratio (ch2_mean / ch1_max) |

## Processing Workflow

1. **Load CZI file**: Read multi-channel microscopy data
2. **Convert to TIFF**: Temporary conversion for Cellpose compatibility
3. **Cell segmentation**: Run Cellpose model on channel 2
4. **Mask separation**: Extract individual cell masks
5. **Measure intensities**: Calculate per-cell statistics for both channels
6. **Export results**: Save mask images and CSV measurements

## Configuration

The script uses `parameters.yaml` for processing parameters:

```yaml
Model Type: "cyto2"           # Cellpose model type
Channels: [2, 3]              # Channel configuration for Cellpose
DNA Gray Thresh: 10           # Threshold parameters
OCR Gray Thresh: 10
Merged Gray Thresh: 10
```

To modify parameters, edit `/mnt/project/parameters.yaml` or update the path in `load_parameters()`.

## Examples

### Example 1: Single File Processing
```bash
python ocicalc.py /data/experiments/sample_001.czi
```

**Output:**
- `/data/experiments/sample_001_mask1.png`
- `/data/experiments/sample_001_mask2.png`
- `/data/experiments/sample_001_cell_wise.csv`

### Example 2: Batch Processing
```bash
python ocicalc.py -f /data/experiments/batch_20241029/
```

Processes all `.czi` files in the folder (non-recursive).

### Example 3: Error Handling
If processing fails for some files, the script:
- Logs detailed error messages
- Continues processing remaining files
- Reports summary statistics at completion

## Architecture

### Key Components

- **`CZIProcessor`**: Main processing class
  - Initializes Cellpose model
  - Handles CZI → TIFF conversion
  - Performs segmentation
  - Calculates measurements

- **`process_file()`**: Per-file processing pipeline
- **`main()`**: CLI argument parsing and orchestration

### Design Decisions

1. **Non-recursive folder processing**: Processes only `.czi` files in the specified folder (not subdirectories)
2. **Same masks for both channels**: Single segmentation applied to both channels
3. **Temporary TIFF conversion**: Uses temp files (auto-deleted) for Cellpose compatibility
4. **Error isolation**: Failed files don't stop batch processing

## Integration with Existing Code

This script **reuses existing project components**:
- ✅ Imports and uses `helper.py` functions (`separator`, `removeZeros`, etc.)
- ✅ Loads configuration from `parameters.yaml`
- ✅ Uses same processing logic as notebook
- ❌ Does NOT duplicate helper code

## Troubleshooting

### Common Issues

**Import Error: `ModuleNotFoundError: No module named 'czifile'`**
```bash
pip install czifile
```

**Import Error: `ModuleNotFoundError: No module named 'helper'`**
- Ensure `helper.py` is in the same directory as `ocicalc.py`, or
- Add the project directory to `PYTHONPATH`:
  ```bash
  export PYTHONPATH="/mnt/project:$PYTHONPATH"
  ```

**Cellpose GPU Error**
- Ensure CUDA is installed and accessible
- Or modify `CZIProcessor.__init__()` to use `gpu=False`

**Parameters File Not Found**
- Update path in `load_parameters()` to point to your `parameters.yaml`

## Performance Notes

- **GPU acceleration**: Enabled by default for Cellpose (requires CUDA)
- **Memory usage**: Proportional to image size and number of detected cells
- **Processing time**: Varies by image size and number of cells (~10-60 seconds per file)

## Limitations

- Processes only `.czi` format (not `.tif`, `.png`, etc.)
- Non-recursive folder search
- Fixed output naming scheme
- Requires Cellpose model files (downloaded automatically on first run)

## Contributing

To extend functionality:
1. Additional measurements → Modify `calculate_measurements()`
2. Different output formats → Modify `save_measurements()` or `save_mask_images()`
3. Custom segmentation → Replace Cellpose in `process_czi()`

## License

Part of the OCICalc project. Refer to project license for terms.

## Contact

For issues or questions, refer to the main project documentation.
