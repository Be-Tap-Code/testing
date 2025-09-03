from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import glob
import random
from typing import List, Dict
from pydantic import BaseModel
import csv
import faiss
import numpy as np
import pandas as pd
import os
import logging
from elasticsearch import Elasticsearch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.getLogger("uvicorn.access").disabled = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Thiết lập cache cho HuggingFace về /data/cndt_hangdv/AIC
os.environ["HF_HOME"] = "/data/cndt_hangdv/AIC"
os.environ["TRANSFORMERS_CACHE"] = "/data/cndt_hangdv/AIC/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/data/cndt_hangdv/AIC/huggingface_datasets"
os.environ["HF_METRICS_CACHE"] = "/data/cndt_hangdv/AIC/huggingface_metrics"
# Force using GPU 5 (which has more free memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
clip_model = None
clip_processor = None
milvus_client = None
DEVICE = None

app = FastAPI(title="Frame Video API", version="1.0.0")

# CORS middleware để frontend có thể kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn đến folder chứa frames
FRAME_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frame")
MODEL_NAME = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
VECTOR_DIMENSION = 1280
FAISS_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_index.faiss"))
METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip_model.eval()
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    df_meta = pd.read_csv(METADATA_PATH)
    image_paths_from_db = df_meta['image_path'].tolist()
except Exception as e:
    print(f"❌ Error initializing model or FAISS/metadata: {e}")
# Model cho response
class FrameData(BaseModel):
    id: str
    filename: str
    video_id: str
    frame_number: int
    image_url: str
    # Optional fields for consistency
    title: str = ""
    duration: str = ""
    vidInfo: str = ""
    timestamp: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0

def parse_filename(filename: str) -> Dict:
    """
    Parse filename theo format: Video-id_frame_hh:mm:ss.ms_ms
    Ví dụ: L03_V001_006000_00-04-00.000_240000.webp
    """
    print(f"🔍 Parsing filename: {filename}")
    
    try:
        # Loại bỏ extension .webp
        name_without_ext = filename.replace('.webp', '')
        print(f"   📝 Name without extension: {name_without_ext}")
        
        # Split theo underscore
        parts = name_without_ext.split('_')
        print(f"   🔪 Split parts: {parts} (length: {len(parts)})")
        
        if len(parts) >= 5:
            level = parts[0]  # L03
            video_id = parts[1]  # V001
            frame_number = parts[2]  # 006000
            timestamp = parts[3]  # 00-04-00.000
            milliseconds = parts[4]  # 240000
            
            print(f"   📊 Parsed components:")
            print(f"      - Level: {level}")
            print(f"      - Video ID: {video_id}")
            print(f"      - Frame Number: {frame_number}")
            try:
                name_without_ext = filename.replace('.webp', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 5:
                    level = parts[0]
                    video_id = parts[1]
                    frame_number = parts[2]
                    timestamp = parts[3]
                    milliseconds = parts[4]
                    formatted_timestamp = timestamp.replace('-', ':')
                    title = f"{level}_{video_id} - Frame {frame_number} - {formatted_timestamp}"
                    duration = formatted_timestamp
                    vid_info = f"*vid: {level}_{video_id} - {milliseconds}ms"
                    result = {
                        "title": title,
                        "duration": duration,
                        "vidInfo": vid_info,
                        "timestamp": formatted_timestamp,
                        "video_id": f"{level}_{video_id}",
                        "frame_number": frame_number,
                        "level": level
                    }
                    return result
                else:
                    raise ValueError(f"Filename format incorrect: expected 5 parts, got {len(parts)}")
            except Exception:
                fallback_result = {
                    "title": filename.replace('.webp', ''),
                    "duration": "00:00:00",
                    "vidInfo": f"*vid: {filename}",
                    "timestamp": "00:00:00",
                    "video_id": "unknown",
                    "frame_number": "0",
                    "level": "unknown"
                }
                return fallback_result
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        clip_model.eval()
        # Load FAISS index và metadata
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        df_meta = pd.read_csv(METADATA_PATH)
        image_paths_from_db = df_meta['image_path'].tolist()
    except Exception as e:
        print(f"❌ Error initializing model or FAISS/metadata: {e}")

# --- Encode text ---
def encode_text(text):
    global clip_model, clip_processor, DEVICE
    if clip_model is None or clip_processor is None or DEVICE is None:
        raise RuntimeError("CLIP model or processor not loaded.")
    import torch
    inputs = clip_processor(text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
        )
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().squeeze().numpy().reshape(1, -1)

# --- Search endpoint ---
class SearchRequest(BaseModel):
    query: str

@app.post("/api/search-frames", response_model=List[FrameData])
async def search_frames(request: SearchRequest, top_k: int = 500):
    try:
        print(f"[API] Nhận query: {request.query}")
        if clip_model is None or clip_processor is None or faiss_index is None or image_paths_from_db is None:
            print("[API] Model hoặc FAISS index/metadata chưa sẵn sàng!")
            raise HTTPException(status_code=500, detail="CLIP model hoặc FAISS index/metadata chưa được khởi tạo. Kiểm tra lại thư viện transformers, torch, faiss, image_index.faiss và image_metadata.csv.")
        query_vector = encode_text(request.query)
        k = min(top_k, len(image_paths_from_db))
        distances, indices = faiss_index.search(query_vector, k)
        indices = indices[0]
        frames = []
        for i, idx in enumerate(indices):
            # if idx < 0 or idx >= len(image_paths_from_db):
            #     continue
            image_path = image_paths_from_db[idx]
            filename = os.path.basename(image_path)
            folder_name = os.path.basename(os.path.dirname(image_path))
            file_path = os.path.join(FRAME_FOLDER, folder_name, filename)
            # if not os.path.exists(file_path):
            #     continue
            meta_row = df_meta[df_meta['image_path'] == image_path]
            # if meta_row.empty:
            #     continue
            video_id = meta_row.iloc[0]['video_name'] if 'video_name' in meta_row.columns else ''
            frame_number = int(meta_row.iloc[0]['frame_number']) if 'frame_number' in meta_row.columns else 0
            frames.append(FrameData(
                id=f"frame-search-{i+1}",
                filename=filename,
                video_id=video_id,
                frame_number=frame_number,
                image_url=f"/api/frames/{folder_name}/{filename}"
            ))
        return frames
    except Exception as e:
        import traceback
        print("❌ Error in /api/search-frames:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Frame Video API is running!"}

@app.get("/api/status")
async def get_status():
    """Get API status and model status"""
    status = {
        "api": "running",
        "clip": {
            "model_loaded": clip_model is not None,
            "processor_loaded": clip_processor is not None,
            "device": DEVICE,
            "metadata_loaded": df_meta is not None,
            "metadata_rows": len(df_meta) if df_meta is not None else 0
        },
        "elasticsearch": {
            "ocr_index": OCR_INDEX,
            "audio_index": ELASTIC_AUDIO_INDEX,
            "connected": es is not None
        }
    }
    return status

@app.get("/api/frames", response_model=List[FrameData])
async def get_frames(limit: int = 2000):
    """
    Lấy danh sách frames với giới hạn số lượng
    """
    try:
        # Lấy tất cả file .webp trong các folder con của FRAME_FOLDER
        pattern = os.path.join(FRAME_FOLDER, "*", "*.webp")
        all_files = glob.glob(pattern)
        if not all_files:
            raise HTTPException(status_code=404, detail="No frame files found")
        # Random select files theo limit
        selected_files = random.sample(all_files, min(limit, len(all_files)))
        frames = []
        for i, file_path in enumerate(selected_files):
            filename = os.path.basename(file_path)
            folder = os.path.basename(os.path.dirname(file_path))
            parsed_data = parse_filename(filename)
            frame_data = FrameData(
                id=f"frame-{i+1}",
                filename=filename,
                title=parsed_data["title"],
                duration=parsed_data["duration"],
                vidInfo=parsed_data["vidInfo"],
                timestamp=parsed_data["timestamp"],
                video_id=parsed_data["video_id"],
                frame_number=int(parsed_data["frame_number"]),
                image_url=f"/api/frames/{folder}/{filename}"
            )
            frames.append(frame_data)
        return frames
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting frames: {str(e)}")

@app.get("/api/frames/{folder}/{filename}")
async def get_frame_image(folder: str, filename: str):
    """
    Serve file ảnh frame
    """
    file_path = os.path.join(FRAME_FOLDER, folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(
        file_path,
        media_type="image/webp",
        headers={"Cache-Control": "max-age=3600"}
    )

# API mapping tới link YouTube chính xác
import json
import re
from urllib.parse import urlparse, parse_qs


# YouTube mapping cache từ file CSV
youtube_mapping_cache = None

def load_youtube_mapping_cache():
    """Load và cache YouTube mapping từ file CSV"""
    global youtube_mapping_cache
    if youtube_mapping_cache is not None:
        return youtube_mapping_cache
    print("🔍 Loading YouTube mapping cache from CSV...")
    youtube_mapping_cache = {}
    mapping_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "youtube_mapping.csv"))
    import csv
    with open(mapping_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            watch_url = row["watch_url"]
            # Extract YouTube video_id from watch_url
            yt_id = None
            if "youtube.com/watch" in watch_url:
                parsed_url = urlparse(watch_url)
                query_params = parse_qs(parsed_url.query)
                yt_id = query_params.get('v', [None])[0]
            elif "youtu.be/" in watch_url:
                yt_id = watch_url.split("youtu.be/")[-1].split("?")[0]
            elif "youtube.com/embed/" in watch_url:
                yt_id = watch_url.split("youtube.com/embed/")[-1].split("?")[0]
            if yt_id:
                youtube_mapping_cache[yt_id] = video_id
    print(f"✅ YouTube mapping cache loaded: {len(youtube_mapping_cache)} entries")
    return youtube_mapping_cache

# API tìm frame dựa trên YouTube URL và thời gian
class SearchFrameRequest(BaseModel):
    youtube_url: str
    seconds: int


@app.post("/api/search-frame-id")
async def search_frame_id(request: SearchFrameRequest):
    """
    Tìm frame dựa trên YouTube URL và thời gian (giây)
    Trả về video_id và frame_number tương ứng
    """
    try:
        print(f"🔍 Search frame request: URL={request.youtube_url}, seconds={request.seconds}")
        # Parse YouTube URL để lấy YouTube video_id
        yt_id = None
        if "youtube.com/watch" in request.youtube_url:
            parsed_url = urlparse(request.youtube_url)
            query_params = parse_qs(parsed_url.query)
            yt_id = query_params.get('v', [None])[0]
        elif "youtu.be/" in request.youtube_url:
            yt_id = request.youtube_url.split("youtu.be/")[-1].split("?")[0]
        elif "youtube.com/embed/" in request.youtube_url:
            yt_id = request.youtube_url.split("youtube.com/embed/")[-1].split("?")[0]
        if not yt_id:
            raise HTTPException(status_code=400, detail="Không thể parse được video ID từ YouTube URL")
        print(f"✅ Parsed YouTube video ID: {yt_id}")
        # Load YouTube mapping cache
        youtube_mapping = load_youtube_mapping_cache()
        # Tìm video_id metadata từ mapping
        if yt_id not in youtube_mapping:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thông tin cho video ID: {yt_id}")
        video_id_metadata = youtube_mapping[yt_id]
        print(f"✅ Found video_id_metadata: {video_id_metadata}")
        # Tìm fps từ metadata
        METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))
        import pandas as pd
        df_meta = pd.read_csv(METADATA_PATH)
        meta_rows = df_meta[df_meta['video_name'] == video_id_metadata]
        if meta_rows.empty:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy metadata cho video: {video_id_metadata}")
        fps = float(meta_rows.iloc[0]['fps'])
        print(f"✅ Found FPS: {fps}")
        # Tính frame_number dựa trên thời gian và fps
        frame_number = int(request.seconds * fps)
        print(f"✅ Calculated frame number: {frame_number} (from {request.seconds}s * {fps}fps)")
        # Tìm frame gần nhất trong metadata
        meta_rows['frame_number'] = meta_rows['frame_number'].astype(int)
        meta_rows['diff'] = (meta_rows['frame_number'] - frame_number).abs()
        nearest_row = meta_rows.loc[meta_rows['diff'].idxmin()]
        actual_frame_number = int(nearest_row['frame_number'])
        actual_seconds = actual_frame_number / fps
        print(f"✅ Nearest frame: {actual_frame_number} (at {actual_seconds:.2f}s)")
        # Trả về kết quả
        result = {
            "youtube_url": request.youtube_url,
            "video_id": yt_id,
            "video_id_metadata": video_id_metadata,
            "requested_seconds": request.seconds,
            "actual_seconds": round(actual_seconds, 2),
            "fps": fps,
            "requested_frame_number": frame_number,
            "actual_frame_number": actual_frame_number
        }
        print(f"🎯 Search frame completed successfully: {result}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error in search_frame_id: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm frame: {str(e)}")


@app.get("/api/youtube-link")
async def get_youtube_link(video_id: str, frame_number: int):
    try:
        # Đọc metadata
        METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))
        df_meta = pd.read_csv(METADATA_PATH)

        # Kiểm tra metadata cho video_id
        meta_row = df_meta[df_meta['video_name'] == video_id]
        if meta_row.empty or 'fps' not in meta_row.columns:
            return {"error": f"Không tìm thấy metadata hoặc fps cho video_id: {video_id}"}

        fps = float(meta_row.iloc[0]['fps'])

        # Lấy watch_url từ youtube_mapping.csv
        mapping_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "youtube_mapping.csv"))
        watch_url = None
        with open(mapping_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video_id"] == video_id:
                    watch_url = row["watch_url"]
                    break

        if not watch_url:
            return {"error": f"Không tìm thấy watch_url cho video_id: {video_id}"}

        # Tính timestamp (giây)
        seconds = int(int(frame_number) / fps) if fps > 0 else 0

        # Build YouTube link with timestamp
        url = f"{watch_url}&t={seconds}s" if "?" in watch_url else f"{watch_url}?t={seconds}s"

        # Tìm frame gần nhất trong metadata
        meta_row = meta_row.copy()
        meta_row['frame_number'] = meta_row['frame_number'].astype(int)
        meta_row['diff'] = (meta_row['frame_number'] - int(frame_number)).abs()
        nearest_row = meta_row.loc[meta_row['diff'].idxmin()]

        filename_real = os.path.basename(nearest_row['image_path'])
        folder = video_id
        image_url = f"/api/frames/{folder}/{filename_real}"

        return {
            "youtube_url": url,
            "timestamp": seconds,
            "fps": fps,
            "image_url": image_url
        }

    except Exception as e:
        return {"error": str(e)}

# API filter frame theo orientation
from fastapi import Query, Body
from fastapi.responses import JSONResponse

class FrameFilterRequest(BaseModel):
    items: list  # [{"video_id": ..., "frame_number": ...}]
    orientation: str

@app.post("/api/filter-frames")
async def filter_frames(request: FrameFilterRequest):
    try:
        orientation = request.orientation
        items = request.items
        filtered = []
        for item in items:
            video_id = item.get("video_id")
            frame_number = item.get("frame_number")
            # Tìm frame trong df_meta
            meta_row = df_meta[(df_meta['video_name'] == video_id) & (df_meta['frame_number'] == int(frame_number))]
            if meta_row.empty:
                continue
            width = int(meta_row.iloc[0]['width'])
            height = int(meta_row.iloc[0]['height'])
            image_path = meta_row.iloc[0]['image_path']
            # Lấy folder và filename từ image_path
            parts = image_path.split('/')
            if len(parts) == 2:
                folder, filename = parts
                image_url = f"/api/frames/{folder}/{filename}"
            else:
                image_url = ""
            # Filter theo orientation
            if orientation == "Ngang" and width > height:
                filtered.append({
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "width": width,
                    "height": height,
                    "image_url": image_url
                })
            elif orientation == "Dọc" and width < height:
                filtered.append({
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "width": width,
                    "height": height,
                    "image_url": image_url
                })
        return filtered
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# --- CẤU HÌNH ELASTICSEARCH ---

# --- CẤU HÌNH ELASTICSEARCH ---
ELASTIC_CLOUD_URL = "https://my-elasticsearch-project-cf1442.es.us-east-1.aws.elastic.cloud:443"
ELASTIC_API_KEY = "cTdNYW5KZ0JpUGhEQ3ItZmdSc2Q6RWRTbHloajF1MGRxTDNVU2txZlZMdw==" # Thay bằng API Key của bạn
OCR_INDEX = "ocr_batch1"  # Đổi tên index cho đúng
es = Elasticsearch(
    ELASTIC_CLOUD_URL,
    api_key=ELASTIC_API_KEY,
    request_timeout=60
)

# --- TẢI DỮ LIỆU METADATA MỘT LẦN KHI ỨNG DỤNG KHỞI ĐỘNG ---
METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))
try:
    DF_META = pd.read_csv(METADATA_PATH)
    DF_META.set_index('image_path', inplace=True)
    print("Đã tải xong image_metadata.csv vào bộ nhớ.")
except Exception as e:
    DF_META = None
    print(f"❌ Lỗi khi tải image_metadata.csv: {e}")

class OCRSearchRequest(BaseModel):
    text: str


@app.post("/api/search_ocr", response_model=List[FrameData])
async def search_ocr(request: OCRSearchRequest):
    # N-gram fuzzy search query
    search_query = {
        "query": {
            "nested": {
                "path": "ocr_entries",
                "query": {
                    "match": {
                        "ocr_entries.text": {
                            "query": request.text,
                            "minimum_should_match": "5%"
                        }
                    }
                },
                "inner_hits": {}
            }
        }
    }

    try:
        res = es.search(index=OCR_INDEX, body=search_query, size=500)
        frames = []
        for idx, hit in enumerate(res["hits"]["hits"]):
            source = hit["_source"]
            image_path = source.get("image_path")
            filename = ""
            folder = ""
            video_id = ""
            frame_number = 0
            title = ""
            duration = ""
            vidInfo = ""
            timestamp = ""
            width = 0
            height = 0
            fps = 0.0
            # Lấy text khớp nhất từ inner_hits
            matched_entry = None
            if "inner_hits" in hit and "ocr_entries" in hit["inner_hits"]:
                inner_hits = hit["inner_hits"]["ocr_entries"]["hits"]["hits"]
                if inner_hits:
                    matched_entry = inner_hits[0]["_source"]
            matched_text = matched_entry["text"] if matched_entry else ""
            matched_coords = matched_entry["coordinate"] if matched_entry else ""
            if image_path:
                parts = image_path.split('/')
                if len(parts) == 2:
                    folder, filename = parts
                    # Ưu tiên lấy metadata từ DF_META
                    if DF_META is not None:
                        try:
                            meta_row = DF_META.loc[image_path]
                            video_id = meta_row['video_name'] if 'video_name' in meta_row else ""
                            frame_number = int(meta_row['frame_number']) if 'frame_number' in meta_row else 0
                            width = int(meta_row['width']) if 'width' in meta_row else 0
                            height = int(meta_row['height']) if 'height' in meta_row else 0
                            fps = float(meta_row['fps']) if 'fps' in meta_row else 0.0
                            # Tính timestamp từ frame_number và fps nếu có
                            if fps > 0:
                                seconds = frame_number / fps
                                timestamp = f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
                                duration = timestamp
                            else:
                                timestamp = ""
                                duration = ""
                            title = f"{video_id} - Frame {frame_number} - {timestamp}"
                            vidInfo = f"*vid: {video_id} - {frame_number} frames @ {fps}fps"
                        except KeyError:
                            # image_path không có trong metadata, fallback parse filename
                            parsed = None
                            try:
                                parsed = parse_filename(filename)
                            except Exception:
                                parsed = None
                            if parsed:
                                video_id = parsed.get("video_id", "")
                                frame_number = int(parsed.get("frame_number", 0))
                                title = parsed.get("title", "")
                                duration = parsed.get("duration", "")
                                vidInfo = parsed.get("vidInfo", "")
                                timestamp = parsed.get("timestamp", "")
                    else:
                        # Không có DF_META, fallback parse filename
                        parsed = None
                        try:
                            parsed = parse_filename(filename)
                        except Exception:
                            parsed = None
                        if parsed:
                            video_id = parsed.get("video_id", "")
                            frame_number = int(parsed.get("frame_number", 0))
                            title = parsed.get("title", "")
                            duration = parsed.get("duration", "")
                            vidInfo = parsed.get("vidInfo", "")
                            timestamp = parsed.get("timestamp", "")
                image_url = f"/api/frames/{folder}/{filename}" if folder and filename else ""
                frames.append(FrameData(
                    id=f"ocr-search-{idx+1}",
                    filename=filename,
                    video_id=video_id,
                    frame_number=frame_number,
                    image_url=image_url,
                    title=title,
                    duration=duration,
                    vidInfo=f"OCR: {matched_text} | {vidInfo}",
                    timestamp=timestamp,
                    width=width,
                    height=height,
                    fps=fps
                ))
        return frames
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NearestFrameRequest(BaseModel):
    video_id: str
    frame_id: int

@app.post("/api/nearest-frame-link")
async def get_nearest_frame_link(request: NearestFrameRequest):
    """
    Trả về link frame gần nhất với frame_id cho video_id.
    """
    try:
        # Đọc metadata
        METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))
        df_meta = pd.read_csv(METADATA_PATH)

        # Lọc các frame thuộc video_id
        meta_rows = df_meta[df_meta['video_name'] == request.video_id]
        if meta_rows.empty:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy metadata cho video_id: {request.video_id}")

        # Tìm frame gần nhất với frame_id
        meta_rows = meta_rows.copy()
        meta_rows['frame_number'] = meta_rows['frame_number'].astype(int)
        meta_rows['diff'] = (meta_rows['frame_number'] - int(request.frame_id)).abs()
        nearest_row = meta_rows.loc[meta_rows['diff'].idxmin()]

        filename_real = os.path.basename(nearest_row['image_path'])
        folder = request.video_id
        image_url = f"/api/frames/{folder}/{filename_real}"
        nearest_frame_number = int(nearest_row['frame_number'])

        return {
            "video_id": request.video_id,
            "requested_frame_id": request.frame_id,
            "nearest_frame_number": nearest_frame_number,
            "image_url": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
ELASTIC_AUDIO_INDEX = "asr_batch1"

class AudioSearchRequest(BaseModel):
    text: str
    size: int = 100

@app.post("/api/search-audio")
async def search_audio(request: AudioSearchRequest):
    # Multi-match n-gram search for transcript and transcript.ngram
    search_query = {
        "size": request.size,
        "query": {
            "multi_match": {
                "query": request.text,
                "fields": ["transcript", "transcript.ngram"],
                "type": "best_fields"
            }
        },
        "highlight": {
            "pre_tags": ["<MARK>"],
            "post_tags": ["</MARK>"],
            "fields": {
                "transcript": {}
            }
        }
    }
    def parse_time_to_seconds(t):
        parts = t.split(":")
        parts = [int(p) for p in parts if p.isdigit()]
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        elif len(parts) == 1:
            return parts[0]
        return 0
    METADATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image_metadata.csv"))
    df_meta = pd.read_csv(METADATA_PATH)
    try:
        res = es.search(index=ELASTIC_AUDIO_INDEX, body=search_query, size=200)
        frames = []
        for idx, hit in enumerate(res["hits"]["hits"]):
            source = hit["_source"]
            video_path = source.get("video_path", "")
            start_time_str = source.get("start_time", "0")
            transcript = source.get("transcript", "")
            highlight = ""
            if "highlight" in hit and "transcript" in hit["highlight"]:
                highlight = " ... ".join(hit["highlight"]["transcript"])
            filename = os.path.basename(video_path)
            video_id = filename.split(".")[0] if filename else ""
            seconds = parse_time_to_seconds(start_time_str)
            meta_row = df_meta[df_meta['video_name'] == video_id]
            if meta_row.empty:
                # Log lỗi hoặc trả về thông báo rõ ràng
                print(f"Không tìm thấy metadata cho video_id: {video_id}")
                continue
            fps = float(meta_row.iloc[0]['fps']) if 'fps' in meta_row.columns and not pd.isnull(meta_row.iloc[0]['fps']) else 25.0
            frame_number = int(seconds * fps)
            meta_video = df_meta[df_meta['video_name'] == video_id]
            if not meta_video.empty:
                meta_video['frame_number'] = meta_video['frame_number'].astype(int)
                meta_video['diff'] = (meta_video['frame_number'] - frame_number).abs()
                nearest_row = meta_video.loc[meta_video['diff'].idxmin()]
                filename_real = os.path.basename(nearest_row['image_path'])
                folder = video_id
                width = int(nearest_row['width']) if 'width' in nearest_row else 0
                height = int(nearest_row['height']) if 'height' in nearest_row else 0
                image_url = f"/api/frames/{folder}/{filename_real}"
            else:
                frame_number_str = str(frame_number).zfill(8)
                filename_real = f"{video_id}_{frame_number_str}.webp"
                folder = video_id
                width = 0
                height = 0
                image_url = f"/api/frames/{folder}/{filename_real}"
            title = f"{video_id} - ASR @ {start_time_str}"
            duration = start_time_str
            vidInfo = f"ASR: {transcript}"
            if highlight:
                vidInfo += f" | Highlight: {highlight}"
            frames.append(FrameData(
                id=f"asr-search-{idx+1}",
                filename=filename_real,
                video_id=video_id,
                frame_number=frame_number,
                image_url=image_url,
                title=title,
                duration=duration,
                vidInfo=vidInfo,
                timestamp=duration,
                width=width,
                height=height,
                fps=fps
            ))
        return frames
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)