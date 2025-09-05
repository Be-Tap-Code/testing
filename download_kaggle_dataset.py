#!/usr/bin/env python3
"""
Script để tải dataset FAISS từ Kaggle và thay thế các file hiện tại
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import zipfile

def install_kaggle():
    """Cài đặt Kaggle API"""
    print("📦 Installing Kaggle API...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "kaggle"], check=True)
        print("✅ Kaggle API installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Kaggle API: {e}")
        return False

def setup_kaggle_credentials():
    """Kiểm tra Kaggle credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        # Đảm bảo quyền 600
        kaggle_file.chmod(0o600)
        print(f"✅ Found kaggle.json at {kaggle_file}")
        return True
    else:
        print(f"❌ kaggle.json not found at {kaggle_file}")
        return False

def download_dataset():
    """Tải dataset từ Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    
    dataset_name = "betapcode/data-faiss-batch-2"  # slug Kaggle dataset
    download_dir = Path("./kaggle_download")
    download_dir.mkdir(exist_ok=True)
    
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(download_dir), "--unzip"]
        print(f"Downloading {dataset_name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Failed to download dataset: {result.stderr}")
            return False
        print("✅ Dataset downloaded and unzipped successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def extract_and_replace_files():
    """Thay thế các file cần thiết"""
    print("\n📂 Replacing files...")
    download_dir = Path("./kaggle_download")
    if not download_dir.exists():
        print("❌ Download directory not found")
        return False

    files_to_replace = {
        "image_index.faiss": "image_index.faiss",
        "image_metadata.csv": "image_metadata.csv"
    }

    for source_name, target_name in files_to_replace.items():
        source_files = list(download_dir.rglob(source_name))
        target_path = Path(target_name)

        if source_files:
            source_file = source_files[0]
            print(f"📄 Replacing {target_name} ...")
            if target_path.exists():
                backup_path = Path(f"{target_name}.backup")
                shutil.move(target_path, backup_path)
                print(f"   Backed up original to {backup_path}")
            shutil.copy2(source_file, target_path)
            print(f"   ✅ Replaced {target_name}")
        else:
            print(f"   ⚠️  {source_name} not found in downloaded files")
    
    print("✅ File replacement completed!")
    return True

def cleanup():
    """Dọn dẹp file tạm"""
    print("\n🧹 Cleaning up temporary files...")
    download_dir = Path("./kaggle_download")
    if download_dir.exists():
        shutil.rmtree(download_dir)
        print("✅ Cleanup completed!")

def main():
    print("🚀 Kaggle FAISS Dataset Downloader")
    print("=" * 50)
    
    if not install_kaggle():
        return
    if not setup_kaggle_credentials():
        print("\n❌ Please setup Kaggle credentials first!")
        return
    if not download_dataset():
        print("\n❌ Failed to download dataset!")
        return
    if not extract_and_replace_files():
        print("\n❌ Failed to replace files!")
        return
    cleanup()
    print("\n🎉 Dataset download and replacement completed successfully!")

if __name__ == "__main__":
    main()
