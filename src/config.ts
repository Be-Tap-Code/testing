// Configuration for the application
export const CONFIG = {
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  DEFAULT_FRAME_LIMIT: 200,
  MAX_RETRY_ATTEMPTS: 3,
} as const;

export const API_ENDPOINTS = {
  FRAMES: '/api/frames',
  FRAME_IMAGE: '/api/frames/image',   // dùng: `${API_BASE_URL}${FRAME_IMAGE}/${filename}`
  FRAME_INFO: '/api/frames/info',     // dùng: `${API_BASE_URL}${FRAME_INFO}/${filename}`

  SEARCH_FRAMES: '/api/search-frames', // CLIP
  SEARCH_BEIT3: '/api/search-beit3',   // BEiT3
  SEARCH_OCR: '/api/search_ocr',       // Elasticsearch OCR
  SEARCH_AUDIO: '/api/search-audio',   // Elasticsearch ASR

  FILTER_FRAMES: '/api/filter-frames', // POST {items, orientation}
  YT_LINK: '/api/youtube-link',        // GET ?video_id=&frame_number=&fps=
  FIND_BY_YOUTUBE: '/api/frames/by-youtube',
} as const;