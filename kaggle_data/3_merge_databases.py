
import os
import shutil

# Danh sách các thư mục chứa frame cần copy
DB_DIRS_TO_MERGE = [
    '/kaggle/input/database-l21-l23/',
    '/kaggle/input/database-l24a-l25b/',
    '/kaggle/input/database-l26a-l26c/',
    '/kaggle/input/database-l26d-l27/',
    '/kaggle/input/database-l28-l30/'
    # Thêm các đường dẫn đến các thư mục database khác vào đây nếu cần
]

if __name__ == "__main__":
    FRAME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frame'))
    print(f"--- ĐANG TỰ ĐỘNG TẢI TẤT CẢ FRAME VÀO THƯ MỤC: {FRAME_DIR} ---")
    for db_dir in DB_DIRS_TO_MERGE:
        src_frame_dir = os.path.join(db_dir, 'frame')
        if os.path.exists(src_frame_dir):
            for folder in os.listdir(src_frame_dir):
                src_folder = os.path.join(src_frame_dir, folder)
                dst_folder = os.path.join(FRAME_DIR, folder)
                if os.path.isdir(src_folder):
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder, exist_ok=True)
                    for file in os.listdir(src_folder):
                        src_file = os.path.join(src_folder, file)
                        dst_file = os.path.join(dst_folder, file)
                        shutil.copy2(src_file, dst_file)
                        print(f"    -> Đã copy: {src_file} -> {dst_file}")
        else:
            print(f"Không tìm thấy thư mục frame trong {db_dir}, bỏ qua.")
    print("--- ĐÃ HOÀN TẤT VIỆC TẢI FRAME ---")
