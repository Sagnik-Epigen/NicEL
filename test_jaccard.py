#!/usr/bin/env python3
"""
Jaccard Index Calculation Demo and Test Script

This script demonstrates and tests the Jaccard index (IoU) calculation
for cell segmentation evaluation. Can be used standalone to test the metric
or as a reference implementation.

Usage:
    # Test with synthetic data
    python test_jaccard.py
    
    # Test with your own masks
    python test_jaccard.py --pred prediction_mask.png --gt groundtruth_mask.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path


def calculate_jaccard_index(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                           verbose: bool = True) -> dict:
    """
    Calculate comprehensive Jaccard index metrics.
    
    Args:
        pred_mask: Predicted segmentation mask (cell IDs)
        gt_mask: Ground truth segmentation mask (cell IDs)
        verbose: Print detailed information
    
    Returns:
        Dictionary with all Jaccard metrics
    """
    # Pixel-level Jaccard (binary masks)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    pixel_jaccard = float(intersection / union) if union > 0 else 0.0
    
    # Instance-level Jaccard (matching individual cells)
    gt_ids = np.unique(gt_mask)[1:]  # Exclude background (0)
    pred_ids = np.unique(pred_mask)[1:]
    
    matches = []
    ious = []
    
    if verbose:
        print("\n" + "="*60)
        print("JACCARD INDEX CALCULATION")
        print("="*60)
    
    for gt_id in gt_ids:
        gt_cell = (gt_mask == gt_id)
        best_iou = 0.0
        best_pred_id = None
        
        for pred_id in pred_ids:
            pred_cell = (pred_mask == pred_id)
            
            cell_intersection = np.logical_and(gt_cell, pred_cell).sum()
            cell_union = np.logical_or(gt_cell, pred_cell).sum()
            
            if cell_union > 0:
                iou = cell_intersection / cell_union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id
        
        ious.append(best_iou)
        if best_iou >= 0.5:  # Standard threshold
            matches.append((gt_id, best_pred_id, best_iou))
            if verbose:
                print(f"Cell {gt_id} matched to Pred {best_pred_id} (IoU: {best_iou:.4f})")
        else:
            if verbose:
                print(f"Cell {gt_id} unmatched (Best IoU: {best_iou:.4f})")
    
    # Calculate metrics
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)
    n_matched = len(matches)
    
    mean_iou = np.mean(ious) if ious else 0.0
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'pixel_jaccard': pixel_jaccard,
        'instance_jaccard': mean_iou,
        'gt_cells': n_gt,
        'pred_cells': n_pred,
        'matched_cells': n_matched,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'per_cell_ious': ious
    }
    
    if verbose:
        print("\n" + "-"*60)
        print("SUMMARY METRICS")
        print("-"*60)
        print(f"Pixel-level Jaccard Index:    {pixel_jaccard:.4f}")
        print(f"Instance-level Jaccard Index: {mean_iou:.4f}")
        print(f"Ground Truth Cells:           {n_gt}")
        print(f"Predicted Cells:              {n_pred}")
        print(f"Matched Cells (IoU ≥ 0.5):    {n_matched}")
        print(f"Precision:                    {precision:.4f}")
        print(f"Recall:                       {recall:.4f}")
        print(f"F1 Score:                     {f1_score:.4f}")
        print("="*60 + "\n")
    
    return results


def create_synthetic_masks(scenario: str = "good") -> tuple:
    """
    Create synthetic ground truth and prediction masks for testing.
    
    Args:
        scenario: 'perfect', 'good', 'moderate', 'poor', 'split', 'merge'
    
    Returns:
        Tuple of (gt_mask, pred_mask, description)
    """
    size = (256, 256)
    gt_mask = np.zeros(size, dtype=np.uint16)
    pred_mask = np.zeros(size, dtype=np.uint16)
    
    if scenario == "perfect":
        # Perfect overlap
        cv.circle(gt_mask, (80, 80), 30, 1, -1)
        cv.circle(gt_mask, (180, 80), 25, 2, -1)
        cv.circle(gt_mask, (130, 160), 35, 3, -1)
        
        pred_mask = gt_mask.copy()
        description = "Perfect segmentation (Jaccard = 1.0)"
    
    elif scenario == "good":
        # Good overlap (~0.85 IoU)
        cv.circle(gt_mask, (80, 80), 30, 1, -1)
        cv.circle(gt_mask, (180, 80), 25, 2, -1)
        cv.circle(gt_mask, (130, 160), 35, 3, -1)
        
        cv.circle(pred_mask, (82, 82), 28, 1, -1)
        cv.circle(pred_mask, (178, 82), 23, 2, -1)
        cv.circle(pred_mask, (128, 158), 33, 3, -1)
        description = "Good segmentation (slight offset)"
    
    elif scenario == "moderate":
        # Moderate overlap (~0.6 IoU)
        cv.circle(gt_mask, (80, 80), 30, 1, -1)
        cv.circle(gt_mask, (180, 80), 25, 2, -1)
        cv.circle(gt_mask, (130, 160), 35, 3, -1)
        
        cv.circle(pred_mask, (85, 85), 25, 1, -1)
        cv.circle(pred_mask, (175, 85), 20, 2, -1)
        cv.circle(pred_mask, (125, 155), 30, 3, -1)
        description = "Moderate segmentation (undersized)"
    
    elif scenario == "poor":
        # Poor overlap, some missed cells
        cv.circle(gt_mask, (80, 80), 30, 1, -1)
        cv.circle(gt_mask, (180, 80), 25, 2, -1)
        cv.circle(gt_mask, (130, 160), 35, 3, -1)
        
        cv.circle(pred_mask, (90, 90), 20, 1, -1)
        cv.circle(pred_mask, (130, 165), 25, 2, -1)
        # Cell 2 missed completely
        description = "Poor segmentation (one cell missed)"
    
    elif scenario == "split":
        # Over-segmentation: one cell split into two
        cv.ellipse(gt_mask, (128, 128), (60, 40), 0, 0, 360, 1, -1)
        
        cv.ellipse(pred_mask, (100, 128), (35, 35), 0, 0, 360, 1, -1)
        cv.ellipse(pred_mask, (156, 128), (35, 35), 0, 0, 360, 2, -1)
        description = "Over-segmentation (cell split)"
    
    elif scenario == "merge":
        # Under-segmentation: two cells merged into one
        cv.circle(gt_mask, (100, 128), 30, 1, -1)
        cv.circle(gt_mask, (156, 128), 30, 2, -1)
        
        cv.ellipse(pred_mask, (128, 128), (60, 40), 0, 0, 360, 1, -1)
        description = "Under-segmentation (cells merged)"
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return gt_mask, pred_mask, description


def visualize_comparison(gt_mask: np.ndarray, pred_mask: np.ndarray, 
                        title: str = "", save_path: str = None):
    """Visualize ground truth vs prediction with overlap."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Ground truth
    axes[0, 0].imshow(gt_mask, cmap=plt.cm.nipy_spectral)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Prediction
    axes[0, 1].imshow(pred_mask, cmap=plt.cm.nipy_spectral)
    axes[0, 1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Overlap visualization
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    
    # Create RGB overlay: green=GT, red=pred, yellow=overlap
    overlap_vis = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    overlap_vis[gt_binary == 1] = [0, 255, 0]  # Green for GT
    overlap_vis[pred_binary == 1] = [255, 0, 0]  # Red for pred
    overlap_vis[np.logical_and(gt_binary, pred_binary)] = [255, 255, 0]  # Yellow for overlap
    
    axes[1, 0].imshow(overlap_vis)
    axes[1, 0].set_title('Overlap (Green=GT, Red=Pred, Yellow=Match)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map
    diff = np.abs(gt_binary.astype(int) - pred_binary.astype(int))
    axes[1, 1].imshow(diff, cmap='Reds')
    axes[1, 1].set_title('Difference Map', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def test_all_scenarios():
    """Test Jaccard calculation on all synthetic scenarios."""
    scenarios = ['perfect', 'good', 'moderate', 'poor', 'split', 'merge']
    
    print("\n" + "="*70)
    print("TESTING JACCARD INDEX ON SYNTHETIC DATA")
    print("="*70)
    
    results_summary = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print('='*70)
        
        gt_mask, pred_mask, description = create_synthetic_masks(scenario)
        print(f"Description: {description}")
        
        results = calculate_jaccard_index(gt_mask, pred_mask, verbose=True)
        results_summary.append({
            'scenario': scenario,
            'description': description,
            **results
        })
        
        # Visualize
        visualize_comparison(gt_mask, pred_mask, 
                           title=f"{scenario.upper()}: {description}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY OF ALL SCENARIOS")
    print("="*70)
    print(f"{'Scenario':<12} {'Pixel IoU':>10} {'Instance IoU':>12} {'Precision':>10} {'Recall':>10}")
    print("-"*70)
    for r in results_summary:
        print(f"{r['scenario']:<12} {r['pixel_jaccard']:>10.4f} {r['instance_jaccard']:>12.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f}")
    print("="*70)


def test_custom_masks(pred_path: str, gt_path: str):
    """Test Jaccard calculation on custom mask files."""
    print(f"\nLoading masks...")
    print(f"  Prediction: {pred_path}")
    print(f"  Ground Truth: {gt_path}")
    
    # Load masks
    pred_mask = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
    gt_mask = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
    
    if pred_mask is None or gt_mask is None:
        print("Error: Could not load one or both masks!")
        return
    
    # Ensure correct data type
    pred_mask = pred_mask.astype(np.uint16)
    gt_mask = gt_mask.astype(np.uint16)
    
    # Calculate metrics
    results = calculate_jaccard_index(gt_mask, pred_mask, verbose=True)
    
    # Visualize
    visualize_comparison(gt_mask, pred_mask, 
                        title=f"Custom Masks Comparison",
                        save_path="jaccard_comparison.png")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Test and demonstrate Jaccard index calculation for cell segmentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test with synthetic data:
    %(prog)s
  
  Test with custom masks:
    %(prog)s --pred prediction.png --gt groundtruth.png
  
  Test specific scenario:
    %(prog)s --scenario good
        """
    )
    
    parser.add_argument(
        '--pred',
        type=str,
        default=None,
        help='Path to prediction mask image'
    )
    
    parser.add_argument(
        '--gt',
        type=str,
        default=None,
        help='Path to ground truth mask image'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['perfect', 'good', 'moderate', 'poor', 'split', 'merge', 'all'],
        default='all',
        help='Test specific synthetic scenario (default: all)'
    )
    
    args = parser.parse_args()
    
    # Custom masks mode
    if args.pred and args.gt:
        if not Path(args.pred).exists():
            print(f"Error: Prediction mask not found: {args.pred}")
            return
        if not Path(args.gt).exists():
            print(f"Error: Ground truth mask not found: {args.gt}")
            return
        
        test_custom_masks(args.pred, args.gt)
    
    # Synthetic data mode
    else:
        if args.scenario == 'all':
            test_all_scenarios()
        else:
            gt_mask, pred_mask, description = create_synthetic_masks(args.scenario)
            print(f"\nScenario: {args.scenario.upper()}")
            print(f"Description: {description}")
            calculate_jaccard_index(gt_mask, pred_mask, verbose=True)
            visualize_comparison(gt_mask, pred_mask, 
                               title=f"{args.scenario.upper()}: {description}")


if __name__ == "__main__":
    main()
