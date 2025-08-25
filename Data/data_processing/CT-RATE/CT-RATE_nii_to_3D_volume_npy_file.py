import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import monai.transforms as mtf
import nibabel as nib
from tqdm import tqdm
import concurrent.futures

'''
Batch process NIfTI files and convert to NPY files, selecting files to process based on metadata
You can use this script to process the CT-RaTE dataset NIfTI files into 3D volume NPY files.
Make sure to follow these steps:
1. change the 'train_base_dirs' and 'valid_base_dirs' to the directories where your NIfTI files are located.
2. change the 'train_metadata_path' and 'valid_metadata_path' to the paths of the metadata CSV files in CT-RaTE dataset. 
3. change the 'train_reports_path' and 'valid_reports_path' to the paths of the radiology text reports CSV files in CT-RaTE dataset.
4. change the 'output_dir' to the directory where you want to save the path of processed NPY files into JSON format.
'''

def resize_array(array, current_spacing, target_spacing):
    """
    Resize array to match target spacing
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array


def nii_img_to_tensor(path, transform, metadata_df):
    """
    Read NIfTI file and convert to tensor
    """
    try:
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return None
            
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()  # (512, 512, 303)
        
        # Get filename
        file_name = os.path.basename(path)
        
        # Find corresponding metadata row
        row = metadata_df[metadata_df['VolumeName'] == file_name]
        if len(row) == 0:
            print(f"Metadata not found for file {file_name}")
            return None
            
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Target spacing
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        # Convert data to PyTorch tensor and move to GPU
        img_data_tensor = torch.tensor(img_data).to('cuda')
        img_data_tensor = slope * img_data_tensor + intercept
        
        # Limit HU value range
        hu_min, hu_max = -1000, 200
        img_data_tensor = torch.clamp(img_data_tensor, min=hu_min, max=hu_max)
        
        # Transfer back to CPU and convert to NumPy array
        img_data = img_data_tensor.cpu().numpy()
        img_data = img_data.transpose(2, 0, 1)
        
        # Convert to tensor and adjust dimensions
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        # Resize
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        # Move data to GPU and convert to float32
        img_data_tensor = torch.tensor(img_data).to('cuda', dtype=torch.float32)
        img_data = img_data_tensor.cpu().numpy()

        tensor = torch.tensor(img_data)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        # Min-max normalization
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        tensor = tensor - min_val
        normalized_tensor = tensor / torch.clamp(max_val - min_val, min=1e-8)

        # Apply transform
        processed_tensor = transform(normalized_tensor)

        return processed_tensor
        
    except Exception as e:
        print(f"Error processing file {path}: {str(e)}")
        return None


# Define image transformation
transform = mtf.Compose([
    mtf.CropForeground(),
    mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
])

def process_single_file(nii_file, output_dir, metadata_df):
    """Process a single NIfTI file and save as NPY"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output file path
        file_name = os.path.basename(nii_file).replace('.nii.gz', '')
        output_path = os.path.join(output_dir, f"{file_name}_3D_features.npy")
        
        # Skip processing if file already exists
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            return nii_file, output_path
        
        # Process NIfTI file
        video_tensor = nii_img_to_tensor(nii_file, transform, metadata_df)
        
        if video_tensor is not None:
            # Save processed tensor
            np.save(output_path, video_tensor.cpu().numpy())
            print(f"Saved: {output_path}")
            return nii_file, output_path
        else:
            print(f"Processing failed: {nii_file}")
            return None
    except Exception as e:
        print(f"Error processing file {nii_file}: {str(e)}")
        return None

def find_nifti_file(volume_name, base_dirs):
    """Find NIfTI file with given volume name"""
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            if volume_name in files:
                return os.path.join(root, volume_name)
    return None

def process_dataset_from_metadata(metadata_path, base_dirs, output_dir, dataset_type, max_rows=None, max_workers=4):
    """Process NIfTI files based on metadata"""
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Read metadata
    try:
        metadata_df = pd.read_csv(metadata_path)
        if max_rows:
            metadata_df = metadata_df.head(max_rows)
        
        print(f"Read {len(metadata_df)} rows of metadata from {metadata_path}")
        
        # Find all NIfTI files
        nii_files = []
        for _, row in tqdm(metadata_df.iterrows(), desc="Searching files", total=len(metadata_df)):
            volume_name = row['VolumeName']
            nii_file = find_nifti_file(volume_name, base_dirs)
            if nii_file:
                nii_files.append(nii_file)
            else:
                print(f"File not found: {volume_name}")
        
        print(f"Found {len(nii_files)} NIfTI files, will process these files")
        
        processed_files = []
        
        # Use thread pool to process files
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, nii_file, dataset_output_dir, metadata_df): nii_file 
                    for nii_file in nii_files}
            
            # Process results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(nii_files), desc="Processing files"):
                nii_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                except Exception as e:
                    print(f"Error occurred while processing file {nii_file}: {str(e)}")
        
        return processed_files
    
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return []

def create_dataset_json(train_processed_files, valid_processed_files, train_reports_path, valid_reports_path, output_json_path):
    """Create dataset JSON file"""
    # Read report data
    train_reports_df = pd.read_csv(train_reports_path)
    valid_reports_df = pd.read_csv(valid_reports_path)
    
    # Create mapping from filename to report
    train_volume_to_findings = {}
    for _, row in train_reports_df.iterrows():
        volume_name = row['VolumeName']
        findings = row['Findings_EN']
        if pd.notna(findings):
            train_volume_to_findings[volume_name] = findings
    
    valid_volume_to_findings = {}
    for _, row in valid_reports_df.iterrows():
        volume_name = row['VolumeName']
        findings = row['Findings_EN']
        if pd.notna(findings):
            valid_volume_to_findings[volume_name] = findings
    
    # Create dataset
    dataset = {"train": [], "validation": []}
    
    # Add training data
    for src, dst in train_processed_files:
        file_name = os.path.basename(src)
        if file_name in train_volume_to_findings:
            entry = {
                "image": dst,
                "text": train_volume_to_findings[file_name]
            }
            dataset["train"].append(entry)
    
    # Add validation data
    for src, dst in valid_processed_files:
        file_name = os.path.basename(src)
        if file_name in valid_volume_to_findings:
            entry = {
                "image": dst,
                "text": valid_volume_to_findings[file_name]
            }
            dataset["validation"].append(entry)
    
    print(f"Dataset size - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")
    
    # Save JSON file
    with open(output_json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset JSON saved to: {output_json_path}")

if __name__ == "__main__":
    # Set paths
    train_base_dirs = ["./CT-RATE/CT-RATE-new/dataset/train"]
    valid_base_dirs = ["./CT-RATE/CT-RATE-new/dataset/valid"]
    output_dir = "./CT-RATE/processed/processed_train_debug"
    train_metadata_path = "./CT-RATE/CT-RATE-new/dataset/metadata/train_metadata.csv"
    valid_metadata_path = "./CT-RATE/CT-RATE-new/dataset/metadata/validation_metadata.csv"
    train_reports_path = "./CT-RATE/CT-RATE-new/dataset/radiology_text_reports/train_reports.csv"
    valid_reports_path = "./CT-RATE/CT-RATE-new/dataset/radiology_text_reports/validation_reports.csv"
    output_json_path = "./CT-RATE/ct_rate_dataset.json"
    
    # Process training data - based on first 500 rows of metadata
    print("Processing training data...")
    train_processed_files = process_dataset_from_metadata(
        train_metadata_path, train_base_dirs, output_dir, 
        dataset_type="train", max_rows=500
    )
    
    # Process validation data - based on first 30 rows of metadata
    print("Processing validation data...")
    valid_processed_files = process_dataset_from_metadata(
        valid_metadata_path, valid_base_dirs, output_dir, 
        dataset_type="validation", max_rows=30
    )
    
    print(f"Processing complete! Training files: {len(train_processed_files)}, Validation files: {len(valid_processed_files)}")
    
    # Save processed file information as JSON
    file_mapping = {
        "train": [{"original": src, "processed": dst} for src, dst in train_processed_files],
        "validation": [{"original": src, "processed": dst} for src, dst in valid_processed_files]
    }
    
    with open(os.path.join(output_dir, "processed_files_mapping.json"), "w") as f:
        json.dump(file_mapping, f, indent=2)
    
    # Create dataset JSON
    print("Creating dataset JSON...")
    create_dataset_json(
        train_processed_files, valid_processed_files, 
        train_reports_path, valid_reports_path, output_json_path
    )