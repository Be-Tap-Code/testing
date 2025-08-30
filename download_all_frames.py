#!/usr/bin/env python3
"""
Script để tải tất cả các frame từ các dataset Kaggle
"""
import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Danh sách các dataset cần tải
DATASETS = [
    {
        "name": "authienan/frame-l21-l23",
        "description": "Frame L21-L23",
        "expected_folders": ["L21", "L22", "L23"]
    },
    {
        "name": "authienan/frame-l24a-l25b", 
        "description": "Frame L24A-L25B",
        "expected_folders": ["L24", "L25"]
    },
    {
        "name": "qhiiine/frame-l26a-l26c",
        "description": "Frame L26A-L26C", 
        "expected_folders": ["L26"]
    },
    {
        "name": "qhiiine/frame-l26d-l27",
        "description": "Frame L26D-L27",
        "expected_folders": ["L26", "L27"]
    },
    {
        "name": "qhiiine/frame-l28-l30",
        "description": "Frame L28-L30",
        "expected_folders": ["L28", "L29", "L30"]
    }
]

def download_dataset(dataset_info):
    """Tải một dataset từ Kaggle"""
    dataset_name = dataset_info["name"]
    description = dataset_info["description"]
    
    print(f"\n📥 Downloading {description} ({dataset_name})...")
    
    # Tạo thư mục tạm cho dataset này
    temp_dir = Path(f"./temp_download_{dataset_name.replace('/', '_')}")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Tải dataset
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(temp_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 phút timeout
        
        if result.returncode != 0:
            print(f"❌ Failed to download {dataset_name}: {result.stderr}")
            return False, dataset_info, temp_dir
        
        print(f"✅ Downloaded {description}")
        return True, dataset_info, temp_dir
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout downloading {dataset_name}")
        return False, dataset_info, temp_dir
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False, dataset_info, temp_dir

def extract_dataset(dataset_info, temp_dir):
    """Giải nén dataset"""
    dataset_name = dataset_info["name"]
    description = dataset_info["description"]
    
    print(f"📦 Extracting {description}...")
    
    try:
        # Tìm file zip
        zip_files = list(temp_dir.glob("*.zip"))
        if not zip_files:
            print(f"❌ No zip file found for {dataset_name}")
            return False
        
        zip_file = zip_files[0]
        
        # Giải nén
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print(f"✅ Extracted {description}")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {dataset_name}: {e}")
        return False

def move_frames_to_destination(dataset_info, temp_dir):
    """Di chuyển frames vào thư mục đích"""
    dataset_name = dataset_info["name"]
    description = dataset_info["description"]
    print(f"📂 Moving frames from {description}...")
    frame_dir = Path("./frame")
    frame_dir.mkdir(exist_ok=True)
    moved_count = 0
    try:
        # Tìm tất cả thư mục video trong temp_dir
        for item in temp_dir.rglob("*"):
            if item.is_dir() and item.name.startswith("L") and "_V" in item.name:
                dest_path = frame_dir / item.name
                if dest_path.exists():
                    print(f"   ⚠️  {item.name} already exists, skipping...")
                    continue
                shutil.move(str(item), str(dest_path))
                moved_count += 1
                print(f"   ✅ Moved {item.name}")
        print(f"✅ Moved {moved_count} video folders from {description}")
        return True
    except Exception as e:
        print(f"❌ Error moving frames from {dataset_name}: {e}")
        return False

def cleanup_temp_dir(temp_dir):
    """Dọn dẹp thư mục tạm"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️  Warning: Could not cleanup {temp_dir}: {e}")

def check_existing_frames():
    """Kiểm tra frames hiện có"""
    print("📊 Checking existing frames...")
    
    frame_dir = Path("./frame")
    if not frame_dir.exists():
        print("❌ Frame directory does not exist")
        return []
    
    existing_videos = []
    for item in frame_dir.iterdir():
        if item.is_dir():
            existing_videos.append(item.name)
    
    existing_videos.sort()
    print(f"📁 Found {len(existing_videos)} existing video folders:")
    for video in existing_videos[:10]:  # Show first 10
        print(f"   - {video}")
    
    if len(existing_videos) > 10:
        print(f"   ... and {len(existing_videos) - 10} more")
    
    return existing_videos

def process_dataset(dataset_info):
    """Xử lý một dataset hoàn chỉnh"""
    dataset_name = dataset_info["name"]
    description = dataset_info["description"]
    
    print(f"\n{'='*60}")
    print(f"🚀 Processing {description}")
    print(f"{'='*60}")
    
    # Bước 1: Tải dataset
    success, dataset_info, temp_dir = download_dataset(dataset_info)
    if not success:
        cleanup_temp_dir(temp_dir)
        return False
    
    # Bước 2: Giải nén
    if not extract_dataset(dataset_info, temp_dir):
        cleanup_temp_dir(temp_dir)
        return False
    
    # Bước 3: Di chuyển frames
    if not move_frames_to_destination(dataset_info, temp_dir):
        cleanup_temp_dir(temp_dir)
        return False
    
    # Bước 4: Dọn dẹp
    cleanup_temp_dir(temp_dir)
    
    print(f"🎉 Completed processing {description}")
    return True

def main():
    print("🚀 AIC25 Frame Dataset Downloader")
    print("=" * 60)
    print("This script will download all frame datasets from Kaggle")
    print("=" * 60)
    
    # Kiểm tra frames hiện có
    existing_frames = check_existing_frames()
    
    print(f"\n📋 Will download {len(DATASETS)} datasets:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"   {i}. {dataset['description']} ({dataset['name']})")
    
    # Xác nhận từ user
    response = input(f"\n❓ Continue with download? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Download cancelled by user")
        return
    
    print(f"\n🚀 Starting download process...")
    start_time = time.time()
    
    successful_downloads = 0
    failed_downloads = 0
    
    # Xử lý từng dataset tuần tự (để tránh quá tải)
    for i, dataset_info in enumerate(DATASETS, 1):
        print(f"\n📊 Progress: {i}/{len(DATASETS)}")
        
        if process_dataset(dataset_info):
            successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"❌ Failed to process {dataset_info['description']}")
    
    # Tổng kết
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 Download process completed!")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful_downloads}/{len(DATASETS)} datasets")
    print(f"❌ Failed: {failed_downloads}/{len(DATASETS)} datasets")
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    
    # Kiểm tra lại frames sau khi tải
    print(f"\n📊 Final frame count:")
    final_frames = check_existing_frames()
    new_frames = len(final_frames) - len(existing_frames)
    print(f"📈 Added {new_frames} new video folders")
    
    if successful_downloads > 0:
        print(f"\n✅ You can now restart the backend server to index the new frames!")

if __name__ == "__main__":
    main()
