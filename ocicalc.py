#!/usr/bin/env python3
"""
OCICalc - Optical Cell Index Calculator

Terminal-runnable script for processing .czi microscopy images to extract
per-cell measurements and generate mask visualizations.

Usage:
    Single file:  python ocicalc.py /path/to/image.czi
    Folder:       python ocicalc.py -f /path/to/folder

Dependencies (install via pip):
    - numpy
    - pandas
    - matplotlib
    - czifile
    - cellpose
    - opencv-python (cv2)
    - PyYAML (yaml)
    - scikit-image (for helper module dependencies)
    - helper.py (project module - must be in Python path)

Output per input file:
    - <filename>_mask1.png: Mask visualization for channel 1
    - <filename>_mask2.png: Mask visualization for channel 2
    - <filename>_cell_wise.csv: Per-cell measurements
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import tempfile
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from czifile import imread as cziread, czi2tif
from cellpose import models
from cellpose.io import imread

import helper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Image export settings
DPI = 600
COLORMAP = plt.cm.nipy_spectral


def load_parameters(params_path: str = "./parameters.yaml") -> dict:
    """Load processing parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        sys.exit(1)


class CZIProcessor:
    """Processes CZI microscopy images for cell segmentation and measurement."""
    
    def __init__(self, params: dict):
        """Initialize processor with parameters and Cellpose model."""
        self.params = params
        logger.info(f"Initializing Cellpose model (type: {params['Model Type']})")
        self.model = models.Cellpose(gpu=True, model_type=params["Model Type"])
    
    def process_czi(self, czi_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Process a single CZI file to extract intensities, masks, and measurements.
        
        Args:
            czi_path: Path to .czi file
            
        Returns:
            Tuple of (intensities, masks, tiff_image, individual_masks)
        """
        logger.info(f"Reading CZI file: {czi_path}")
        
        # Read CZI intensities
        czi_intensities = cziread(str(czi_path))
        czi_intensities = np.squeeze(czi_intensities)
        czi_intensities = np.moveaxis(czi_intensities, 0, -1)
        
        # Convert to TIFF for Cellpose processing
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_tif_path = tmp.name
        
        try:
            czi2tif(str(czi_path), tmp_tif_path)
            tiff_img = imread(tmp_tif_path)
            tiff_img = np.moveaxis(tiff_img, 0, -1)
        finally:
            Path(tmp_tif_path).unlink(missing_ok=True)
        
        # Run Cellpose on channel 2 (index 1)
        logger.info("Running Cellpose segmentation...")
        masks, _, _, _ = self.model.eval(
            tiff_img[:, :, 1],
            channels=self.params["Channels"],
            diameter=None
        )
        
        # Separate individual cell masks
        individual_masks = helper.separator(masks)
        logger.info(f"Detected {len(individual_masks)} cells")
        
        return czi_intensities, masks, tiff_img, individual_masks
    
    def calculate_measurements(
        self, 
        czi_intensities: np.ndarray, 
        individual_masks: List[np.ndarray]
    ) -> pd.DataFrame:
        """
        Calculate per-cell measurements from intensities and masks.
        
        Args:
            czi_intensities: Raw intensity data (H x W x C)
            individual_masks: List of binary masks for each cell
            
        Returns:
            DataFrame with per-cell measurements
        """
        logger.info("Calculating per-cell measurements...")
        
        measurements = []
        
        for cell_id, mask in enumerate(individual_masks, start=1):
            # Extract intensities for this cell
            ch1_masked = np.multiply(czi_intensities[:, :, 0], mask.astype(int))
            ch2_masked = np.multiply(czi_intensities[:, :, 1], mask.astype(int))
            
            # Remove zeros (background)
            ch1_values = helper.removeZeros(ch1_masked.flatten())
            ch2_values = helper.removeZeros(ch2_masked.flatten())
            
            # Calculate statistics
            measurements.append({
                'cell_id': cell_id,
                'channel1_mean': np.mean(ch1_values) if len(ch1_values) > 0 else 0,
                'channel2_mean': np.mean(ch2_values) if len(ch2_values) > 0 else 0,
                'channel1_max': np.max(ch1_values) if len(ch1_values) > 0 else 0,
                'channel2_max': np.max(ch2_values) if len(ch2_values) > 0 else 0,
                'channel1_median': np.median(ch1_values) if len(ch1_values) > 0 else 0,
                'channel2_median': np.median(ch2_values) if len(ch2_values) > 0 else 0,
                'area_channel1_mask': np.sum(mask),
                'area_channel2_mask': np.sum(mask),  # Same mask for both channels
                'oci_ratio': (np.max(ch1_values) / np.min(ch2_values) 
                             if len(ch1_values) > 0 and len(ch2_values) > 0 and np.max(ch1_values) > 0 
                             else np.nan)
            })
        #oci=mean(measurements['oci_ratio'])
        #measurements['oci']=oci
        
        return pd.DataFrame(measurements)
    
    def save_mask_images(self, masks: np.ndarray, output_prefix: str) -> None:
        """
        Save mask visualizations as PNG files.
        
        Args:
            masks: Combined mask array
            output_prefix: Output file path without extension
        """
        logger.info("Generating mask visualizations...")
        
        # Create visualizations for both channels (same masks)
        for channel_idx in [1, 2]:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(masks, cmap=COLORMAP)
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            output_path = f"{output_prefix}_mask{channel_idx}.png"
            fig.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            logger.info(f"Saved mask: {output_path}")
    
    def save_measurements(self, df: pd.DataFrame, output_path: str) -> None:
        """Save measurements DataFrame to CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Saved measurements: {output_path}")


def process_file(czi_path: Path, processor: CZIProcessor) -> bool:
    """
    Process a single CZI file and save outputs.
    
    Args:
        czi_path: Path to .czi file
        processor: CZIProcessor instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {czi_path.name}")
        logger.info(f"{'='*60}")
        
        # Process the CZI file
        czi_intensities, masks, tiff_img, individual_masks = processor.process_czi(czi_path)
        
        # Calculate measurements
        measurements_df = processor.calculate_measurements(czi_intensities, individual_masks)
        measurements_df['oci'] = measurements_df['oci_ratio'].mean() 
        
        # Prepare output paths
        output_prefix = str(czi_path.parent / czi_path.stem)
        csv_path = f"{output_prefix}_cell_wise.csv"
        
        # Save outputs
        processor.save_mask_images(masks, output_prefix)
        processor.save_measurements(measurements_df, csv_path)
        
        logger.info(f"✓ Successfully processed {czi_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {czi_path.name}: {str(e)}", exc_info=True)
        return False


def find_czi_files(folder_path: Path) -> List[Path]:
    """
    Find all .czi files in a folder (non-recursive).
    
    Args:
        folder_path: Path to folder
        
    Returns:
        List of .czi file paths
    """
    czi_files = list(folder_path.glob("*.czi"))
    logger.info(f"Found {len(czi_files)} .czi files in {folder_path}")
    return sorted(czi_files)


def validate_input(path: Path, is_folder: bool) -> bool:
    """
    Validate input path and file type.
    
    Args:
        path: Input path
        is_folder: Whether path should be a folder
        
    Returns:
        True if valid, False otherwise
    """
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return False
    
    if is_folder:
        if not path.is_dir():
            logger.error(f"Path is not a folder: {path}")
            return False
    else:
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if path.suffix.lower() != '.czi':
            logger.error(f"File is not a .czi file: {path}")
            return False
    
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process .czi microscopy images for cell segmentation and measurement.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:  %(prog)s /path/to/image.czi
  Folder:       %(prog)s -f /path/to/folder
        """
    )
    
    parser.add_argument(
        'path',
        type=str,
        help='Path to .czi file or folder containing .czi files'
    )
    
    parser.add_argument(
        '-f', '--folder',
        action='store_true',
        help='Treat path as folder and process all .czi files within (non-recursive)'
    )
    
    args = parser.parse_args()
    
    # Validate and prepare input
    input_path = Path(args.path).resolve()
    
    if not validate_input(input_path, args.folder):
        sys.exit(1)
    
    # Load parameters and initialize processor
    params = load_parameters()
    processor = CZIProcessor(params)
    
    # Collect files to process
    if args.folder:
        files_to_process = find_czi_files(input_path)
        if not files_to_process:
            logger.error("No .czi files found in folder")
            sys.exit(1)
    else:
        files_to_process = [input_path]
    
    # Process all files
    logger.info(f"\nProcessing {len(files_to_process)} file(s)...")
    
    success_count = 0
    failure_count = 0
    
    for czi_file in files_to_process:
        if process_file(czi_file, processor):
            success_count += 1
        else:
            failure_count += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed:     {failure_count}")
    logger.info(f"Total:      {len(files_to_process)}")
    
    sys.exit(0 if failure_count == 0 else 1)


if __name__ == "__main__":
    main()
