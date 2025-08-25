import os
import zipfile
import glob
from tqdm import tqdm
import concurrent.futures
import shutil
import argparse


'''
The function extract_zip extracts a single ZIP file to a specified directory, optionally deleting the ZIP file after extraction.
The function extract_all_zips extracts all ZIP files in a given directory using parallel threads, checking
Steps:
1. Set 'dir' to the dataset directory.
2. Set 'workers' to the number of parallel threads.
3. Set 'delete' to True if you want to delete ZIP files after extraction.
4. Set 'organize' to True if you want to organize the dataset directory structure after
'''
def extract_zip(zip_path, extract_dir=None, delete_after=False):
    """
    Extract a single ZIP file
    
    Parameters:
    - zip_path: Path to the ZIP file
    - extract_dir: Target extraction directory, if None, will extract to a directory with the same name as the ZIP file
    - delete_after: Whether to delete the ZIP file after extraction
    
    Returns:
    - True if successful, False if failed
    """
    try:
        # Determine extraction target directory
        if extract_dir is None:
            # Use ZIP filename (without extension) as extraction directory
            extract_dir = os.path.splitext(zip_path)[0]
        
        # Create extraction directory
        os.makedirs(extract_dir, exist_ok=True)
        
        # Get ZIP file size for progress display
        zip_size = os.path.getsize(zip_path)
        
        # Extract files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all files in the archive
            all_files = zip_ref.namelist()
            # Use tqdm to display extraction progress for a single ZIP file
            for file in tqdm(all_files, desc=f"Extracting {os.path.basename(zip_path)}", leave=False):
                zip_ref.extract(file, extract_dir)
        
        # If specified, delete ZIP file after extraction
        if delete_after:
            os.remove(zip_path)
            
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}")
        return False

def extract_all_zips(dataset_dir, max_workers=4, delete_after=False):
    """
    Extract all ZIP files in a directory
    
    Parameters:
    - dataset_dir: Dataset root directory
    - max_workers: Maximum number of parallel worker threads
    - delete_after: Whether to delete ZIP files after extraction
    """
    # Find all ZIP files
    zip_pattern = os.path.join(dataset_dir, "**", "*.zip")
    zip_files = glob.glob(zip_pattern, recursive=True)
    
    print(f"Found {len(zip_files)} ZIP files")
    
    # Check disk space
    statvfs = os.statvfs(dataset_dir)
    free_space = statvfs.f_frsize * statvfs.f_bavail
    total_zip_size = sum(os.path.getsize(zip_file) for zip_file in zip_files)
    
    # Assume extracted size is 5 times the compressed size (this is an estimate)
    estimated_extracted_size = total_zip_size * 5
    
    print(f"Total ZIP file size: {total_zip_size / (1024**3):.2f} GB")
    print(f"Estimated extracted size: {estimated_extracted_size / (1024**3):.2f} GB")
    print(f"Available disk space: {free_space / (1024**3):.2f} GB")
    
    if free_space < estimated_extracted_size:
        print(f"Warning: Disk space may not be sufficient!")
        proceed = input("Continue? (y/n): ")
        if proceed.lower() != 'y':
            print("Operation cancelled")
            return
    
    # Use thread pool for parallel extraction
    success_count = 0
    failed_zips = []
    
    print(f"Starting extraction using {max_workers} parallel threads...")
    
    # For better progress display control, submit tasks individually instead of using map
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_zip = {executor.submit(extract_zip, zip_file, None, delete_after): zip_file for zip_file in zip_files}
        
        # Handle completed tasks
        for future in tqdm(concurrent.futures.as_completed(future_to_zip), total=len(zip_files), desc="Overall progress"):
            zip_file = future_to_zip[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    failed_zips.append(zip_file)
            except Exception as e:
                print(f"Exception occurred while processing {zip_file}: {str(e)}")
                failed_zips.append(zip_file)
    
    print(f"\nExtraction complete!")
    print(f"Successful: {success_count}/{len(zip_files)}")
    
    if failed_zips:
        print(f"Failed: {len(failed_zips)}")
        print("Failed ZIP files:")
        for zip_file in failed_zips:
            print(f"  - {zip_file}")
        
        # Save failed list to file
        with open("failed_zips.txt", "w") as f:
            for zip_file in failed_zips:
                f.write(f"{zip_file}\n")
        print("List of failed ZIP files saved to failed_zips.txt")

def organize_dataset(dataset_dir):
    """
    Organize data according to MedFinder's expected directory structure
    
    Parameters:
    - dataset_dir: Dataset root directory
    """
    print("Starting to organize dataset directory structure...")
    
    # Create CT and meta directories
    ct_base_dir = os.path.join(dataset_dir, "CT")
    meta_dir = os.path.join(dataset_dir, "meta")
    os.makedirs(ct_base_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    
    # Find and move metadata files to meta directory
    metadata_pattern = os.path.join(dataset_dir, "**", "*.csv")
    metadata_files = glob.glob(metadata_pattern, recursive=True)
    
    for meta_file in metadata_files:
        # Ignore files already in meta directory
        if os.path.dirname(meta_file) == meta_dir:
            continue
            
        # Move to meta directory
        target_path = os.path.join(meta_dir, os.path.basename(meta_file))
        shutil.move(meta_file, target_path)
        print(f"Moving metadata: {meta_file} -> {target_path}")
    
    # Find CT-related directories and organize in CT1/ct/file.nii.gz format
    nii_pattern = os.path.join(dataset_dir, "**", "*.nii.gz")
    nii_files = glob.glob(nii_pattern, recursive=True)
    
    for idx, nii_file in enumerate(tqdm(nii_files, desc="Organizing CT files")):
        # Ignore files already in correct location
        if f"/CT" in nii_file and "/ct/" in nii_file:
            continue
            
        # Calculate target CT directory number
        ct_num = (idx // 500) + 1  # Every 500 files in one CT directory
        ct_dir = os.path.join(ct_base_dir, f"CT{ct_num}", "ct")
        os.makedirs(ct_dir, exist_ok=True)
        
        # Move file
        target_path = os.path.join(ct_dir, os.path.basename(nii_file))
        shutil.move(nii_file, target_path)
    
    print("Dataset directory structure organization complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ZIP files from the BIMCV-R dataset")
    parser.add_argument("--dir", type=str, default="./BIMCV_R/dataset", 
                        help="Dataset directory path")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel extraction threads")
    parser.add_argument("--delete", action="store_true", 
                        help="Delete ZIP files after extraction")
    parser.add_argument("--organize", action="store_true", 
                        help="Organize data according to MedFinder's expected directory structure after extraction")
    
    args = parser.parse_args()
    
    # Perform extraction
    extract_all_zips(args.dir, args.workers, args.delete)
    
    # If needed, organize dataset directory structure
    if args.organize:
        organize_dataset(args.dir)