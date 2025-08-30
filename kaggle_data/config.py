# config.py
import torch
import os

# --- CÁC ĐƯỜNG DẪN VÀ CẤU HÌNH CHUNG ---

# Đường dẫn đến thư mục chứa DATABASE TỔNG HỢP đã được gộp lại
# Đây là kết quả của script 3_merge_databases.py
DATABASE_DIR = "./output" 

# Tên các file database bên trong thư mục trên
METADATA_CSV = os.path.join(DATABASE_DIR, "image_metadata.csv")
INDEX_FILE = os.path.join(DATABASE_DIR, "image_index.faiss")

# Tên file submission cuối cùng (nếu có)
SUBMISSION_ZIP = os.path.join(".", "submission.zip")


# --- CẤU HÌNH MODEL VÀ THỰC THI ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
BATCH_SIZE = 32  
RANDOM_SEED = 42

# --- CẤU HÌNH DEBUG/TEST ---
TEST_ON_SMALL_DATASET = False
NUM_TEST_IMAGES = 500
NUM_TEST_QUERIES = 10
