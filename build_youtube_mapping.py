import os
import csv
import json

info_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "info_data"))
output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "youtube_mapping.csv"))

rows = []
for root, dirs, files in os.walk(info_data_dir):
    for json_file in files:
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(root, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            watch_url = info.get('watch_url', '')
            parent_folder = os.path.basename(os.path.dirname(json_path))
            base_id = json_file.replace('.json', '')
            if base_id.startswith(f"{parent_folder}_"):
                video_id = base_id
            else:
                video_id = f"{parent_folder}_{base_id}"
            # Xóa tiền tố "info_data_" nếu có
            if video_id.startswith("info_data_"):
                video_id = video_id.replace("info_data_", "", 1)
            if watch_url:
                rows.append([video_id, watch_url])
        except Exception:
            continue

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video_id', 'watch_url'])
    writer.writerows(rows)

print(f"✅ Saved mapping to {output_csv} ({len(rows)} rows)")