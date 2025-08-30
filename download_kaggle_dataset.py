#!/usr/bin/env python3
"""
Script để tải dataset từ Kaggle và thay thế các file hiện tại
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_kaggle():
    """Cài đặt Kaggle API"""
    print("📦 Installing Kaggle API...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        print("✅ Kaggle API installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Kaggle API: {e}")
        return False

def setup_kaggle_credentials():
    """Hướng dẫn setup Kaggle credentials"""
    print("\n🔑 Kaggle API Setup Required:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token' to download kaggle.json")
    print("3. Place kaggle.json in one of these locations:")
    print("   - ~/.kaggle/kaggle.json (Linux/Mac)")
    print("   - C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
    print("   - Or set KAGGLE_CONFIG_DIR environment variable")
    print("\n4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        print(f"✅ Found kaggle.json at {kaggle_file}")
        return True
    else:
        print(f"❌ kaggle.json not found at {kaggle_file}")
        print("\nAlternatively, you can set environment variables:")
        print("export KAGGLE_USERNAME=your_username")
        print("export KAGGLE_KEY=your_api_key")
        return False

def download_dataset():
    """Tải dataset từ Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    
    # Dataset URL: https://www.kaggle.com/code/phamhoanglenguyen/clip-concat-aic25/output
    # Cần tìm dataset name chính xác
    dataset_name = "phamhoanglenguyen/clip-concat-aic25"
    
    try:
        # Tạo thư mục tạm để tải dataset
        download_dir = Path("./kaggle_download")
        download_dir.mkdir(exist_ok=True)
        
        print(f"Downloading {dataset_name}...")
        
        # Thử tải dataset
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(download_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to download dataset: {result.stderr}")
            print("Trying alternative approach...")
            
            # Thử tải từ competition output
            cmd = ["kaggle", "competitions", "download", "-c", "clip-concat-aic25", "-p", str(download_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Alternative download failed: {result.stderr}")
                return False
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def extract_and_replace_files():
    """Giải nén và thay thế các file"""
    print("\n📂 Extracting and replacing files...")
    
    download_dir = Path("./kaggle_download")
    
    if not download_dir.exists():
        print("❌ Download directory not found")
        return False
    
    try:
        # Tìm file zip
        zip_files = list(download_dir.glob("*.zip"))
        if not zip_files:
            print("❌ No zip files found in download directory")
            return False
        
        zip_file = zip_files[0]
        print(f"📦 Extracting {zip_file.name}...")
        
        # Giải nén
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        # Tìm và thay thế các file cần thiết
        files_to_replace = {
            "image_index.faiss": "image_index.faiss",
            "image_metadata.csv": "image_metadata.csv",
            "keyframe_part_0.index": "keyframe_part_0.index", 
            "keyframe_part_1.index": "keyframe_part_1.index",
            "l27.npy": "l27.npy"
        }
        
        for source_name, target_name in files_to_replace.items():
            # Tìm file trong thư mục giải nén
            source_files = list(download_dir.rglob(source_name))
            target_path = Path(target_name)
            
            if source_files:
                source_file = source_files[0]
                print(f"📄 Replacing {target_name}...")
                
                # Backup file cũ
                if target_path.exists():
                    backup_path = Path(f"{target_name}.backup")
                    shutil.move(target_path, backup_path)
                    print(f"   Backed up original to {backup_path}")
                
                # Copy file mới
                shutil.copy2(source_file, target_path)
                print(f"   ✅ Replaced {target_name}")
            else:
                print(f"   ⚠️  {source_name} not found in downloaded files")
        
        print("✅ File replacement completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting files: {e}")
        return False

def cleanup():
    """Dọn dẹp file tạm"""
    print("\n🧹 Cleaning up temporary files...")
    download_dir = Path("./kaggle_download")
    if download_dir.exists():
        shutil.rmtree(download_dir)
        print("✅ Cleanup completed!")

def main():
    print("🚀 Kaggle Dataset Downloader for AIC25")
    print("=" * 50)
    
    # Bước 1: Cài đặt Kaggle API
    if not install_kaggle():
        return
    
    # Bước 2: Kiểm tra credentials
    if not setup_kaggle_credentials():
        print("\n❌ Please setup Kaggle credentials first!")
        return
    
    # Bước 3: Tải dataset
    if not download_dataset():
        print("\n❌ Failed to download dataset!")
        return
    
    # Bước 4: Giải nén và thay thế
    if not extract_and_replace_files():
        print("\n❌ Failed to extract and replace files!")
        return
    
    # Bước 5: Dọn dẹp
    cleanup()
    
    print("\n🎉 Dataset download and replacement completed successfully!")
    print("You can now restart the backend server to use the new dataset.")

if __name__ == "__main__":
    main()
