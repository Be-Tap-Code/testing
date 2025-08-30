#!/usr/bin/env python3
"""
Script để setup và tải dataset từ Kaggle
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def install_kaggle_api():
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
    """Setup Kaggle credentials"""
    print("\n🔑 Setting up Kaggle credentials...")
    
    # Kiểm tra xem đã có credentials chưa
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        print(f"✅ Found existing kaggle.json at {kaggle_file}")
        return True
    
    print("❌ Kaggle credentials not found!")
    print("\n📋 To setup Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download 'kaggle.json' file")
    print("5. Move the file to ~/.kaggle/kaggle.json")
    print("6. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    print(f"\n💡 Expected location: {kaggle_file}")
    
    # Tạo thư mục .kaggle nếu chưa có
    kaggle_dir.mkdir(exist_ok=True)
    
    # Hỏi user có muốn nhập credentials thủ công không
    print("\n🔧 Alternative: Enter credentials manually")
    username = input("Enter Kaggle username (or press Enter to skip): ").strip()
    
    if username:
        key = input("Enter Kaggle API key: ").strip()
        
        if username and key:
            # Tạo kaggle.json
            credentials = {
                "username": username,
                "key": key
            }
            
            with open(kaggle_file, 'w') as f:
                json.dump(credentials, f)
            
            # Set permissions
            os.chmod(kaggle_file, 0o600)
            
            print(f"✅ Credentials saved to {kaggle_file}")
            return True
    
    return False

def download_kaggle_dataset():
    """Tải dataset từ Kaggle"""
    print("\n📥 Downloading dataset from Kaggle...")
    
    # Thử các cách khác nhau để tải dataset
    commands_to_try = [
        # Thử tải từ notebook output
        ["kaggle", "kernels", "output", "phamhoanglenguyen/clip-concat-aic25", "-p", "./kaggle_data"],
        # Thử tải từ dataset (nếu có)
        ["kaggle", "datasets", "download", "-d", "phamhoanglenguyen/clip-concat-aic25", "-p", "./kaggle_data"],
    ]
    
    for i, cmd in enumerate(commands_to_try, 1):
        print(f"\n🔄 Attempt {i}: {' '.join(cmd)}")
        
        try:
            # Tạo thư mục output
            os.makedirs("./kaggle_data", exist_ok=True)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Download successful!")
                print(f"Output: {result.stdout}")
                return True
            else:
                print(f"❌ Command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def list_downloaded_files():
    """Liệt kê các file đã tải"""
    print("\n📂 Checking downloaded files...")
    
    kaggle_data_dir = Path("./kaggle_data")
    if not kaggle_data_dir.exists():
        print("❌ No kaggle_data directory found")
        return []
    
    files = []
    for file_path in kaggle_data_dir.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            rel_path = file_path.relative_to(kaggle_data_dir)
            files.append((str(rel_path), size_mb))
            print(f"📄 {rel_path}: {size_mb:.1f} MB")
    
    return files

def main():
    print("🚀 Kaggle Dataset Setup for AIC25")
    print("=" * 50)
    
    # Bước 1: Cài đặt Kaggle API
    if not install_kaggle_api():
        print("❌ Cannot proceed without Kaggle API")
        return
    
    # Bước 2: Setup credentials
    if not setup_kaggle_credentials():
        print("❌ Cannot proceed without Kaggle credentials")
        print("\n💡 Please setup credentials manually and run again")
        return
    
    # Bước 3: Tải dataset
    if download_kaggle_dataset():
        print("\n🎉 Dataset download completed!")
        
        # Liệt kê files
        files = list_downloaded_files()
        
        if files:
            print(f"\n📊 Downloaded {len(files)} files")
            print("Now you can run download_dataset_simple.py to replace the current files")
        else:
            print("⚠️  No files found in download directory")
    else:
        print("\n❌ Failed to download dataset")
        print("💡 You may need to:")
        print("   1. Check if the Kaggle notebook/dataset is public")
        print("   2. Verify your Kaggle credentials")
        print("   3. Try downloading manually from the web interface")

if __name__ == "__main__":
    main()
