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
    # {"name": "uiter22521498/frame-k01", "description": "Frame K01", "expected_folders": ["K01"]},
    # {"name": "authienan/frame-k02", "description": "Frame K02", "expected_folders": ["K02"]},
    # {"name": "phamhoanglenguyen/frame-k03", "description": "Frame K03", "expected_folders": ["K03"]},
    # {"name": "uiter22521498/frame-k04", "description": "Frame K04", "expected_folders": ["K04"]},
    # {"name": "authienan/frame-k05", "description": "Frame K05", "expected_folders": ["K05"]},
    # {"name": "authienan/frame-k06", "description": "Frame K06", "expected_folders": ["K06"]},
    # {"name": "authienan/frame-k07", "description": "Frame K07", "expected_folders": ["K07"]},
    # {"name": "uiter22521498/frame-k08", "description": "Frame K08", "expected_folders": ["K08"]},
    # {"name": "uiter22521498/frame-k09", "description": "Frame K09", "expected_folders": ["K09"]},
    # {"name": "authienan/frame-k10", "description": "Frame K10", "expected_folders": ["K10"]},
    # {"name": "authienan/frame-k11", "description": "Frame K11", "expected_folders": ["K11"]},
    {"name": "qhiiine/frame-k12", "description": "Frame K12", "expected_folders": ["K12"]},
    {"name": "qhiiine/frame-k13", "description": "Frame K13", "expected_folders": ["K13"]},
    # {"name": "qhiiine/frame-k14", "description": "Frame K14", "expected_folders": ["K14"]},
    # {"name": "qhiiine/frame-k15", "description": "Frame K15", "expected_folders": ["K15"]},
    # {"name": "authienan/frame-k16", "description": "Frame K16", "expected_folders": ["K16"]},
    # {"name": "qhiiine/frame-k17", "description": "Frame K17", "expected_folders": ["K17"]},
    # {"name": "nguynphmhongl/frame-k18", "description": "Frame K18", "expected_folders": ["K18"]},
    # {"name": "nguynphmhongl/frame-k19", "description": "Frame K19", "expected_folders": ["K19"]},
    # {"name": "qhiiine/frame-k20", "description": "Frame K20", "expected_folders": ["K20"]},
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
            if item.is_dir() and item.name.startswith("K") and "_V" in item.name:
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
    # Ghi log ra file và vẫn in ra màn hình nếu có terminal
    log_file = "download.log"
    class Tee:
        def __init__(self, filename, mode="a"):
            self.file = open(filename, mode, encoding="utf-8")
            self.stdout = sys.stdout
            self.isatty = sys.stdout.isatty()
        def write(self, data):
            self.file.write(data)
            self.file.flush()
            if self.isatty:
                self.stdout.write(data)
                self.stdout.flush()
        def flush(self):
            self.file.flush()
            if self.isatty:
                self.stdout.flush()
        def close(self):
            self.file.close()
    sys.stdout = Tee(log_file, "a")
    sys.stderr = sys.stdout

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
    if not sys.stdin.isatty():
        response = 'y'
    else:
        response = input(f"\n❓ Continue with download? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Download cancelled by user")
        sys.stdout.close()
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
    sys.stdout.close()

if __name__ == "__main__":
    main()
