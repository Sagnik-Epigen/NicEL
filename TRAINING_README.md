# Cellpose Fine-tuning and Evaluation Guide

This guide explains how to use `train_cellpose.py` to fine-tune and evaluate the Cellpose model on your custom CZI microscopy images.

## Overview

The `train_cellpose.py` script provides:
- **Fine-tuning**: Adapt the pretrained Cellpose cyto2 model to your specific cell types and imaging conditions
- **Evaluation**: Calculate Jaccard index (IoU) metrics to measure segmentation performance
- **Data Augmentation**: Automatically augment limited training data with rotations, flips, and intensity variations
- **Visualization**: Generate comparison images of ground truth vs predictions

## Key Metrics

### Jaccard Index (IoU)
The Jaccard index measures the overlap between predicted and ground truth masks:
- **Pixel-level Jaccard**: Overall segmentation quality (0 to 1, where 1 is perfect)
- **Instance-level Jaccard**: Per-cell matching accuracy
- **Precision**: Proportion of predicted cells that match ground truth
- **Recall**: Proportion of ground truth cells that are detected

## File Organization

### Required Structure
```
your_project/
├── train_cellpose.py          # Training script
├── helper.py                   # Helper functions (from your project)
├── parameters.yaml             # Model parameters
├── requirements.txt            # Dependencies
├── images/                     # Your training images
│   ├── sample1.czi
│   ├── sample2.czi
│   └── ...
└── masks/                      # Ground truth masks
    ├── sample1.png             # Mask for sample1.czi
    ├── sample2.png             # Mask for sample2.czi
    └── ...
```

### Ground Truth Mask Requirements
- **Format**: PNG, TIFF, or NPY files
- **Content**: Integer mask where:
  - 0 = background
  - 1, 2, 3, ... = individual cell IDs
- **Naming**: Masks should match image filenames (e.g., `sample1.czi` → `sample1.png`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Basic Training and Evaluation
Train with default parameters (80/20 train/val split, 100 epochs):
```bash
python train_cellpose.py --images ./images --masks ./masks
```

### 2. Custom Training Parameters
Fine-tune with more epochs and lower learning rate:
```bash
python train_cellpose.py \
    --images ./images \
    --masks ./masks \
    --epochs 200 \
    --lr 0.0005 \
    --split 0.75
```

### 3. Disable Data Augmentation
If you have many images and don't need augmentation:
```bash
python train_cellpose.py \
    --images ./images \
    --masks ./masks \
    --no-augment
```

### 4. Evaluation Only (No Training)
Evaluate a pretrained or fine-tuned model:
```bash
python train_cellpose.py \
    --images ./test_images \
    --masks ./test_masks \
    --eval-only \
    --model ./training_output/models/cellpose_finetuned
```

### 5. Custom Output Directory
Specify where to save results:
```bash
python train_cellpose.py \
    --images ./images \
    --masks ./masks \
    --output ./my_training_results
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--images` | Yes | - | Directory containing training images (.czi) |
| `--masks` | Yes | - | Directory containing ground truth masks |
| `--output` | No | `./training_output` | Output directory for models and results |
| `--params` | No | `parameters.yaml` | Path to parameters YAML file |
| `--split` | No | `0.8` | Train/validation split ratio |
| `--epochs` | No | `100` | Number of training epochs |
| `--lr` | No | `0.001` | Learning rate |
| `--weight-decay` | No | `0.0001` | Weight decay for regularization |
| `--no-augment` | No | False | Disable data augmentation |
| `--eval-only` | No | False | Only evaluate, do not train |
| `--model` | No | None | Path to model (for eval-only mode) |

## Output Files

After training/evaluation, the script generates:

```
training_output/
├── models/
│   └── cellpose_finetuned      # Fine-tuned model
├── visualizations/
│   ├── sample1_comparison.png  # Visual comparison for each image
│   ├── sample2_comparison.png
│   └── ...
└── evaluation_results.csv      # Metrics for all images
```

### evaluation_results.csv
Contains per-image metrics:
- `image`: Image filename
- `pixel_jaccard`: Pixel-level Jaccard index
- `instance_jaccard`: Instance-level Jaccard index
- `gt_cells`: Number of ground truth cells
- `pred_cells`: Number of predicted cells
- `matched_cells`: Number of correctly matched cells
- `precision`: Detection precision
- `recall`: Detection recall

## Data Augmentation Strategy

For small datasets, the script automatically applies:
- **Rotations**: 90°, 180°, 270°
- **Flips**: Horizontal and vertical
- **Brightness**: ±20% intensity variations

This increases training data by ~8x.

## Tips for Best Results

### 1. Data Preparation
- Ensure masks accurately label individual cells with unique IDs
- Match mask filenames to image filenames
- Use consistent image quality across your dataset

### 2. Training Parameters
- **Small dataset (<10 images)**: Use augmentation, train for 100-200 epochs
- **Medium dataset (10-50 images)**: Augment if needed, 50-100 epochs
- **Large dataset (>50 images)**: May skip augmentation, 30-50 epochs

### 3. Learning Rate
- Start with default (0.001)
- If training is unstable, reduce to 0.0005 or 0.0001
- If training is too slow, increase to 0.002

### 4. Validation Split
- **Very small datasets (<5 images)**: Use 0.9 or leave-one-out
- **Small datasets (5-20 images)**: Use 0.8
- **Larger datasets**: Use 0.7-0.8

## Interpreting Results

### Good Performance
- Pixel Jaccard > 0.8
- Instance Jaccard > 0.7
- Precision and Recall > 0.85

### Moderate Performance
- Pixel Jaccard: 0.6-0.8
- Instance Jaccard: 0.5-0.7
- May need more training data or epochs

### Poor Performance
- Pixel Jaccard < 0.6
- Instance Jaccard < 0.5
- Consider:
  - Adding more training data
  - Checking mask quality
  - Adjusting model parameters
  - Using different base model

## Troubleshooting

### Issue: "No matching image-mask pairs found"
- **Solution**: Ensure mask filenames match image stems (e.g., `img.czi` → `img.png`)
- Remove suffixes like `_mask`, `_gt`, `_groundtruth` from mask filenames

### Issue: Training crashes or runs out of memory
- **Solution**: 
  - Reduce batch size (edit script to add `batch_size` parameter)
  - Use fewer augmentations (`--no-augment`)
  - Train on smaller image crops

### Issue: Poor validation performance
- **Solution**:
  - Increase training epochs
  - Reduce learning rate
  - Add more diverse training images
  - Check ground truth mask quality

### Issue: Model underfitting
- **Solution**:
  - Train for more epochs
  - Increase learning rate slightly
  - Ensure training data diversity

### Issue: Model overfitting
- **Solution**:
  - Increase weight decay
  - Add more augmentation
  - Reduce training epochs
  - Add more training data

## Integration with Existing Pipeline

To use your fine-tuned model with the main `ocicalc.py` script:

1. After training, note the model path (e.g., `./training_output/models/cellpose_finetuned`)
2. Modify `parameters.yaml`:
   ```yaml
   Model Type: "./training_output/models/cellpose_finetuned"
   ```
3. Run your analysis as normal:
   ```bash
   python ocicalc.py your_image.czi
   ```

## Advanced Usage

### Custom Evaluation Metrics
Edit the `evaluate_model()` function to add:
- Dice coefficient
- Hausdorff distance
- F1 score
- Custom per-cell metrics

### Different Base Models
Change the base model in `parameters.yaml`:
```yaml
Model Type: "cyto"    # For general cells
# or
Model Type: "nuclei"  # For nuclei-specific
```

## Further Reading

- [Cellpose Documentation](https://cellpose.readthedocs.io/)
- [Cellpose Paper](https://www.nature.com/articles/s41592-020-01018-x)
- [Training Custom Models](https://cellpose.readthedocs.io/en/latest/models.html#training-models)

## Support

For issues or questions:
1. Check the terminal output for detailed error messages
2. Verify your ground truth masks are correctly formatted
3. Ensure all dependencies are properly installed
4. Review the visualization outputs to debug segmentation issues
