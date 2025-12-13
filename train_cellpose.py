#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2 as cv

from cellpose import models, io, train
from cellpose.io import imread
import torch
import helper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_parameters(params_path: str = "parameters.yaml") -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded parameters from {params_path}")
        return params
    except FileNotFoundError:
        logger.warning(f"Parameters file {params_path} not found, using defaults")
        return {
            'Model Type': 'cyto2',
            'Channels': [2, 3]
        }


def find_image_files(directory: Path, extensions: List[str] = ['.czi']) -> List[Path]:
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
    return sorted(files)


def find_mask_files(directory: Path, extensions: List[str] = ['.png', '.tif', '.tiff', '.npy']) -> List[Path]:
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
    return sorted(files)


def load_image(image_path: Path) -> np.ndarray:
    try:
        if image_path.suffix.lower() == '.czi':
            import czifile
            with czifile.CziFile(str(image_path)) as czi:
                img_array = czi.asarray()
                while img_array.ndim > 3:
                    img_array = np.squeeze(img_array)
                
                if img_array.ndim == 4:
                    img_array = img_array[0]
                
                if img_array.ndim == 3 and img_array.shape[-1] > 3:
                    img_array = img_array[..., :3]
                elif img_array.ndim == 3 and img_array.shape[0] < img_array.shape[-1]:
                    img_array = np.transpose(img_array, (1, 2, 0))
                
                return img_array.astype(np.float32)
        else:
            return imread(str(image_path))
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_mask(mask_path: Path) -> np.ndarray:
    try:
        if mask_path.suffix.lower() == '.npy':
            mask = np.load(str(mask_path))
        else:
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        
        return mask.astype(np.uint16)
    except Exception as e:
        logger.error(f"Error loading mask {mask_path}: {e}")
        return None


def match_images_and_masks(image_paths: List[Path], mask_paths: List[Path]) -> List[Tuple[Path, Path]]:
    pairs = []
    
    for img_path in image_paths:
        img_stem = img_path.stem
        
        mask_match = None
        for mask_path in mask_paths:
            mask_stem = mask_path.stem
            mask_stem_clean = mask_stem.replace('_mask', '').replace('_gt', '').replace('_groundtruth', '')
            
            if img_stem == mask_stem_clean or img_stem == mask_stem:
                mask_match = mask_path
                break
        
        if mask_match:
            pairs.append((img_path, mask_match))
            logger.debug(f"Matched: {img_path.name} -> {mask_match.name}")
        else:
            logger.warning(f"No mask found for image: {img_path.name}")
    
    return pairs


def calculate_jaccard_index(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def calculate_instance_jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                               threshold: float = 0.5) -> Tuple[float, int, int]:
    gt_ids = np.unique(gt_mask)[1:]  
    pred_ids = np.unique(pred_mask)[1:]
    
    if len(gt_ids) == 0:
        return 0.0, 0, 0
    
    matches = []
    
    for gt_id in gt_ids:
        gt_cell = (gt_mask == gt_id)
        best_iou = 0.0
        
        for pred_id in pred_ids:
            pred_cell = (pred_mask == pred_id)
            
            intersection = np.logical_and(gt_cell, pred_cell).sum()
            union = np.logical_or(gt_cell, pred_cell).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
        
        if best_iou >= threshold:
            matches.append(best_iou)
    
    mean_jaccard = np.mean(matches) if matches else 0.0
    return mean_jaccard, len(matches), len(gt_ids)


def augment_data(images: List[np.ndarray], masks: List[np.ndarray], 
                 n_augmentations: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    aug_images = []
    aug_masks = []
    
    for img, mask in zip(images, masks):
        aug_images.append(img)
        aug_masks.append(mask)
        
        for angle in [90, 180, 270]:
            k = angle // 90
            aug_images.append(np.rot90(img, k))
            aug_masks.append(np.rot90(mask, k))
        
        aug_images.append(np.fliplr(img))
        aug_masks.append(np.fliplr(mask))
        
        aug_images.append(np.flipud(img))
        aug_masks.append(np.flipud(mask))
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            bright = np.clip(img * 1.2, 0, img.max())
            dark = np.clip(img * 0.8, 0, img.max())
        else:
            bright = np.clip(img.astype(float) * 1.2, 0, 255).astype(img.dtype)
            dark = np.clip(img.astype(float) * 0.8, 0, 255).astype(img.dtype)
        
        aug_images.append(bright)
        aug_masks.append(mask.copy())
        
        aug_images.append(dark)
        aug_masks.append(mask.copy())
    
    logger.info(f"Augmented {len(images)} images to {len(aug_images)} images")
    return aug_images, aug_masks


def train_test_split(pairs: List[Tuple[Path, Path]], 
                     split_ratio: float = 0.8) -> Tuple[List, List]:
    n_total = len(pairs)
    n_train = max(1, int(n_total * split_ratio))
    
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    
    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:]]
    
    logger.info(f"Split: {len(train_pairs)} training, {len(val_pairs)} validation")
    return train_pairs, val_pairs


def prepare_training_data(pairs: List[Tuple[Path, Path]], 
                         augment: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    images = []
    masks = []
    
    for img_path, mask_path in pairs:
        img = load_image(img_path)
        mask = load_mask(mask_path)
        
        if img is not None and mask is not None:
            images.append(img)
            masks.append(mask)
        else:
            logger.warning(f"Skipping pair: {img_path.name}")
    
    if augment and len(images) > 0:
        images, masks = augment_data(images, masks)
    
    return images, masks


def train_cellpose_model(train_images: List[np.ndarray], 
                        train_masks: List[np.ndarray],
                        val_images: List[np.ndarray],
                        val_masks: List[np.ndarray],
                        params: dict,
                        output_dir: Path,
                        n_epochs: int = 100,
                        learning_rate: float = 0.001,
                        weight_decay: float = 0.0001) -> models.CellposeModel:
    logger.info("="*60)
    logger.info("STARTING CELLPOSE FINE-TUNING")
    logger.info("="*60)
    
    model_type = params.get('Model Type', 'cyto2')
    channels = params.get('Channels', [2, 3])
    
    logger.info(f"Base model: {model_type}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Training samples: {len(train_images)}")
    logger.info(f"Validation samples: {len(val_images)}")
    logger.info(f"Epochs: {n_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    
    model = models.CellposeModel(gpu=torch.cuda.is_available(), 
                                 model_type=model_type)
    
    train_data = list(zip(train_images, train_masks))
    val_data = list(zip(val_images, val_masks)) if val_images else None
    
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nStarting training...")
    
    try:
        model_path = train.train_seg(
            model.net,
            train_data=train_images,
            train_labels=train_masks,
            test_data=val_images if val_images else None,
            test_labels=val_masks if val_masks else None,
            channels=channels,
            save_path=str(model_dir),
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_name='cellpose_finetuned'
        )
        
        logger.info(f"Model saved to: {model_path}")
        
        trained_model = models.CellposeModel(gpu=torch.cuda.is_available(), 
                                            model_type=str(model_path))
        
        return trained_model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Returning pretrained model instead")
        return model


def evaluate_model(model: models.CellposeModel,
                  test_pairs: List[Tuple[Path, Path]],
                  params: dict,
                  output_dir: Path) -> pd.DataFrame:
    logger.info("="*60)
    logger.info("EVALUATING MODEL")
    logger.info("="*60)
    
    results = []
    channels = params.get('Channels', [2, 3])
    
    for img_path, mask_path in test_pairs:
        logger.info(f"Evaluating: {img_path.name}")
        
        img = load_image(img_path)
        gt_mask = load_mask(mask_path)
        
        if img is None or gt_mask is None:
            logger.warning(f"Skipping {img_path.name}")
            continue
        
        try:
            pred_mask, flows, styles, diams = model.eval(
                img, 
                diameter=None, 
                channels=channels
            )
            
            pixel_jaccard = calculate_jaccard_index(pred_mask, gt_mask)
            instance_jaccard, n_matches, n_gt = calculate_instance_jaccard(
                pred_mask, gt_mask, threshold=0.5
            )
            
            n_pred_cells = len(np.unique(pred_mask)) - 1
            
            results.append({
                'image': img_path.name,
                'pixel_jaccard': pixel_jaccard,
                'instance_jaccard': instance_jaccard,
                'gt_cells': n_gt,
                'pred_cells': n_pred_cells,
                'matched_cells': n_matches,
                'precision': n_matches / n_pred_cells if n_pred_cells > 0 else 0,
                'recall': n_matches / n_gt if n_gt > 0 else 0
            })
            
            logger.info(f"  Pixel Jaccard: {pixel_jaccard:.4f}")
            logger.info(f"  Instance Jaccard: {instance_jaccard:.4f}")
            logger.info(f"  Cells - GT: {n_gt}, Pred: {n_pred_cells}, Matched: {n_matches}")
            
            save_evaluation_visualization(img, gt_mask, pred_mask, img_path.stem, output_dir)
            
        except Exception as e:
            logger.error(f"Evaluation failed for {img_path.name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Mean Pixel Jaccard: {results_df['pixel_jaccard'].mean():.4f} ± {results_df['pixel_jaccard'].std():.4f}")
        logger.info(f"Mean Instance Jaccard: {results_df['instance_jaccard'].mean():.4f} ± {results_df['instance_jaccard'].std():.4f}")
        logger.info(f"Mean Precision: {results_df['precision'].mean():.4f}")
        logger.info(f"Mean Recall: {results_df['recall'].mean():.4f}")
        
        results_path = output_dir / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to: {results_path}")
    
    return results_df


def save_evaluation_visualization(img: np.ndarray, 
                                 gt_mask: np.ndarray, 
                                 pred_mask: np.ndarray,
                                 name: str,
                                 output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if img.ndim == 3:
        axes[0].imshow(img)
    else:
        axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap=plt.cm.nipy_spectral)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap=plt.cm.nipy_spectral)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    save_path = vis_dir / f"{name}_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune and evaluate Cellpose model on custom microscopy data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train and evaluate:
    %(prog)s --images /path/to/images --masks /path/to/masks
  
  Custom training parameters:
    %(prog)s --images /path/to/images --masks /path/to/masks --epochs 200 --lr 0.0005
  
  Evaluation only:
    %(prog)s --images /path/to/images --masks /path/to/masks --eval-only --model /path/to/model
        """
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing training images (.czi files)'
    )
    
    parser.add_argument(
        '--masks',
        type=str,
        required=True,
        help='Directory containing ground truth masks'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./training_output',
        help='Output directory for models and results (default: ./training_output)'
    )
    
    parser.add_argument(
        '--params',
        type=str,
        default='parameters.yaml',
        help='Path to parameters YAML file (default: parameters.yaml)'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train/validation split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--lr',
        '--learning-rate',
        type=float,
        default=0.001,
        dest='learning_rate',
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='Weight decay for regularization (default: 0.0001)'
    )
    
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate, do not train (requires --model)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to pretrained model for evaluation-only mode'
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    params = load_parameters(args.params)
    
    logger.info("Finding and matching image-mask pairs...")
    image_files = find_image_files(images_dir)
    mask_files = find_mask_files(masks_dir)
    
    logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    pairs = match_images_and_masks(image_files, mask_files)
    
    if not pairs:
        logger.error("No matching image-mask pairs found!")
        sys.exit(1)
    
    logger.info(f"Matched {len(pairs)} image-mask pairs")
    
    if args.eval_only:
        if args.model:
            logger.info(f"Loading model from: {args.model}")
            model = models.CellposeModel(gpu=torch.cuda.is_available(), 
                                        model_type=args.model)
        else:
            logger.info("Using pretrained model for evaluation")
            model_type = params.get('Model Type', 'cyto2')
            model = models.CellposeModel(gpu=torch.cuda.is_available(), 
                                        model_type=model_type)
        
        evaluate_model(model, pairs, params, output_dir)
        sys.exit(0)
    
    train_pairs, val_pairs = train_test_split(pairs, args.split)
    
    logger.info("\nPreparing training data...")
    train_images, train_masks = prepare_training_data(
        train_pairs, 
        augment=not args.no_augment
    )
    
    logger.info("Preparing validation data...")
    val_images, val_masks = prepare_training_data(
        val_pairs, 
        augment=False
    )
    
    if not train_images:
        logger.error("No training data loaded!")
        sys.exit(1)
    
    model = train_cellpose_model(
        train_images, train_masks,
        val_images, val_masks,
        params, output_dir,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if val_pairs:
        logger.info("\nEvaluating on validation set...")
        evaluate_model(model, val_pairs, params, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
