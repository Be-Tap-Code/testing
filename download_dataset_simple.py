#!/usr/bin/env python3
"""
Script đơn giản để tải dataset và thay thế files
"""
import os
import requests
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urlparse

def download_file(url, filename):
    """Tải file từ URL"""
    print(f"📥 Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def manual_download_instructions():
    """Hướng dẫn tải thủ công"""
    print("📋 Manual Download Instructions:")
    print("=" * 50)
    print("1. Go to: https://www.kaggle.com/code/phamhoanglenguyen/clip-concat-aic25/output")
    print("2. Click on 'Download' button to download the output files")
    print("3. Extract the downloaded zip file")
    print("4. Copy these files to the project directory:")
    print("   - image_index.faiss")
    print("   - image_metadata.csv") 
    print("   - keyframe_part_0.index")
    print("   - keyframe_part_1.index")
    print("   - l27.npy")
    print("\n5. Or run this script after placing the zip file in current directory")

def find_and_extract_zip():
    """Tìm và giải nén file zip trong thư mục hiện tại"""
    print("\n🔍 Looking for zip files in current directory...")
    
    zip_files = list(Path(".").glob("*.zip"))
    if not zip_files:
        print("❌ No zip files found in current directory")
        return False
    
    print(f"📦 Found zip files: {[f.name for f in zip_files]}")
    
    # Sử dụng file zip đầu tiên
    zip_file = zip_files[0]
    print(f"📂 Extracting {zip_file.name}...")
    
    try:
        extract_dir = Path("./extracted_dataset")
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("✅ Extraction completed!")
        return extract_dir
        
    except Exception as e:
        print(f"❌ Failed to extract zip file: {e}")
        return False

def replace_dataset_files(extract_dir):
    """Thay thế các file dataset"""
    print("\n🔄 Replacing dataset files...")
    
    files_to_replace = {
        "image_index.faiss": "image_index.faiss",
        "image_metadata.csv": "image_metadata.csv", 
        "keyframe_part_0.index": "keyframe_part_0.index",
        "keyframe_part_1.index": "keyframe_part_1.index",
        "l27.npy": "l27.npy"
    }
    
    replaced_count = 0
    
    for source_name, target_name in files_to_replace.items():
        # Tìm file trong thư mục giải nén
        source_files = list(extract_dir.rglob(source_name))
        target_path = Path(target_name)
        
        if source_files:
            source_file = source_files[0]
            print(f"📄 Found {source_name} at {source_file}")
            
            # Backup file cũ nếu tồn tại
            if target_path.exists():
                backup_path = Path(f"{target_name}.backup")
                shutil.move(target_path, backup_path)
                print(f"   📋 Backed up original to {backup_path}")
            
            # Copy file mới
            shutil.copy2(source_file, target_path)
            print(f"   ✅ Replaced {target_name}")
            replaced_count += 1
        else:
            print(f"   ⚠️  {source_name} not found in extracted files")
    
    print(f"\n✅ Replaced {replaced_count}/{len(files_to_replace)} files")
    return replaced_count > 0

def cleanup_temp_files():
    """Dọn dẹp file tạm"""
    print("\n🧹 Cleaning up temporary files...")
    extract_dir = Path("./extracted_dataset")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
        print("✅ Temporary files cleaned up!")

def check_current_files():
    """Kiểm tra các file hiện tại"""
    print("📊 Current dataset files:")
    print("-" * 30)
    
    files_to_check = [
        "image_index.faiss",
        "image_metadata.csv",
        "keyframe_part_0.index", 
        "keyframe_part_1.index",
        "l27.npy"
    ]
    
    for filename in files_to_check:
        path = Path(filename)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"❌ {filename}: Not found")

def main():
    print("🚀 AIC25 Dataset Downloader")
    print("=" * 50)
    
    # Kiểm tra file hiện tại
    check_current_files()
    
    print("\n" + "=" * 50)
    
    # Hướng dẫn tải thủ công
    manual_download_instructions()
    
    print("\n" + "=" * 50)
    
    # Thử tìm và giải nén file zip
    extract_dir = find_and_extract_zip()
    
    if extract_dir:
        # Thay thế files
        if replace_dataset_files(extract_dir):
            print("\n🎉 Dataset replacement completed successfully!")
            
            # Kiểm tra lại files
            print("\n" + "=" * 50)
            check_current_files()
            
            # Dọn dẹp
            cleanup_temp_files()
            
            print("\n✅ You can now restart the backend server to use the new dataset!")
        else:
            print("\n❌ No files were replaced. Please check the zip file contents.")
    else:
        print("\n💡 Please download the dataset manually and place the zip file in this directory, then run this script again.")

if __name__ == "__main__":
    main()
