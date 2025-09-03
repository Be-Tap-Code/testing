  import React, { useCallback, useEffect, useMemo, useState } from 'react';
  import { CONFIG, API_ENDPOINTS } from './config';

  // ============== NEAREST FRAME SEARCH ==============


interface FrameData {
  id: string;
  filename: string;
  video_id: string;
  frame_number: number;   // number như bạn yêu cầu
  image_url: string;

  // optional cho các mode khác
  title?: string;
  duration?: string;
  vidInfo?: string;
  timestamp?: string;
  fps?: number;

  // optional cho filter-frames
  width?: number;
  height?: number;
}

const ensureLeadingSlash = (p: string) => (p.startsWith('/') ? p : '/' + p);
const buildImageUrlFromPath = (image_path: string) =>
  ensureLeadingSlash(`/api/frames/image/${image_path}`);

// Thêm 'youtube' vào SearchMode
type SearchMode = 'random' | 'text'  | 'ocr' | 'audio' | 'youtube'| 'video_frame';
type Orientation = 'All' | 'Ngang' | 'Dọc' | 'Khác';

const SearchResults: React.FC = () => {
  const searchNearestFrame = useCallback(async (videoId: string, frameId: number) => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching nearest frame...');
      const url = `${CONFIG.API_BASE_URL}/api/nearest-frame-link?video_id=${encodeURIComponent(videoId)}&frame_id=${frameId}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`GET /api/nearest-frame-link -> ${res.status}`);
      const data = await res.json();

      // Build frame data
      const frame: FrameData = {
        id: `nearest-${data.video_id}-${data.nearest_frame_number}`,
        filename: `${data.video_id}_${String(data.nearest_frame_number).padStart(8, '0')}.webp`,
        video_id: data.video_id,
        frame_number: data.nearest_frame_number,
        image_url: data.image_url ?? `/api/frames/${data.video_id}/${String(data.nearest_frame_number).padStart(8, '0')}.webp`,
        title: `Nearest frame to ${data.requested_frame_id}`,
        timestamp: String(data.nearest_frame_number),
      };
      setFrames([frame]);
      setAllFrames([frame]);
      setSearchStatus(`Found nearest frame: #${data.nearest_frame_number} for requested #${data.requested_frame_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to search nearest frame');
      setSearchStatus('Nearest frame search failed.');
    } finally {
      setLoading(false);
    }
  }, []);
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [allFrames, setAllFrames] = useState<FrameData[]>([]); // Store original frames for filtering
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  // YouTube inputs
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [videoId, setVideoId] = useState('');
  const [frameId, setFrameId] = useState('');
  const [youtubeSeconds, setYoutubeSeconds] = useState('');

  const [mode, setMode] = useState<SearchMode>('random');
  const [orientation, setOrientation] = useState<Orientation>('All');

  const [searchText, setSearchText] = useState('');
  const [searchStatus, setSearchStatus] = useState('');

  // ============== API CALLS ==============
  const fetchFrames = useCallback(async (isRetry = false) => {
    try {
      setLoading(true);
      setError(null);
      const url = `${CONFIG.API_BASE_URL}${API_ENDPOINTS.FRAMES}?limit=${CONFIG.DEFAULT_FRAME_LIMIT}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`GET ${API_ENDPOINTS.FRAMES} -> ${res.status}`);
      const data: FrameData[] = await res.json();
      setAllFrames(data);
      setFrames(data);
      setRetryCount(0);
      setSearchStatus(`Loaded ${data.length} random frames`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to fetch frames';
      setError(msg);
      if (!isRetry && retryCount < CONFIG.MAX_RETRY_ATTEMPTS) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          fetchFrames(true);
        }, 2000 * (retryCount + 1));
      }
    } finally {
      setLoading(false);
    }
  }, [retryCount]);

  const searchByText = useCallback(async (query: string) => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching with CLIP ...');
      const url = `${CONFIG.API_BASE_URL}${API_ENDPOINTS.SEARCH_FRAMES}`;
      const res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) throw new Error(`POST ${API_ENDPOINTS.SEARCH_FRAMES} -> ${res.status}`);
      const data: FrameData[] = await res.json();
      setAllFrames(data);
      setFrames(data);
      setSearchStatus(`Found ${data.length} frame(s)`);
      setRetryCount(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to search frames';
      setError(msg);
      setSearchStatus('Search failed.');
    } finally {
      setLoading(false);
    }
  }, []);


  // Helper dùng chung cho normalize
  const normalizeToFrameData = (arr: any[]): FrameData[] => {
    return (arr ?? []).map((d: any) => {
      // Nếu là kết quả ASR thô (có video_path/start_time) thì map về schema chuẩn trước
      if (d?.video_path !== undefined || d?.start_time !== undefined) {
        const base = String(d.video_path ?? '').split(/[\\/]/).pop() || '';
        const maybeVideoId = base.replace(/\.[^.]+$/, ''); // bỏ đuôi .mp4
        const start = Math.max(0, Math.floor(Number(d.start_time ?? 0)));
        d = {
          id: d.id ?? `asr-${maybeVideoId}-${start}`,
          filename: `${maybeVideoId}_${start}.webp`,
          video_id: maybeVideoId,
          frame_number: start,
          image_path: `${maybeVideoId}_${start}.webp`,
          title: `${maybeVideoId} - ASR @ ${start}s`,
          duration: String(start),
          vidInfo: d.highlight ?? d.transcript ?? '',
          timestamp: String(start),
          ...d,
        };
      }

      const frame_number =
        typeof d.frame_number === 'number'
          ? d.frame_number
          : Number(d.frame_number ?? d.timestamp ?? 0);

      const image_url =
        d.image_url ??
        (d.image_path ? buildImageUrlFromPath(d.image_path) : `/api/frames/image/${d.video_id ?? ''}_${frame_number}.webp`);

      return {
        ...d,
        frame_number,
        image_url,
      } as FrameData;
    }) as FrameData[];
  };

  const searchByYouTube = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching frame by YouTube...');
      const url = `${CONFIG.API_BASE_URL}/api/search-frame-id`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          youtube_url: youtubeUrl,
          seconds: Number(youtubeSeconds),
        }),
      });
      if (!res.ok) throw new Error(`POST /api/search-frame-id -> ${res.status}`);
      const data = await res.json();

      // Call /api/youtube-link for actual frame
      const actualLinkRes = await fetch(`${CONFIG.API_BASE_URL}/api/youtube-link?video_id=${data.video_id_metadata}&frame_number=${data.actual_frame_number}`);
      const actualLink = actualLinkRes.ok ? await actualLinkRes.json() : {};

      // Call /api/youtube-link for requested frame
      const requestedLinkRes = await fetch(`${CONFIG.API_BASE_URL}/api/youtube-link?video_id=${data.video_id_metadata}&frame_number=${data.requested_frame_number}`);
      const requestedLink = requestedLinkRes.ok ? await requestedLinkRes.json() : {};

      const framesRaw = [
        {
          id: `yt-${data.video_id}-${data.actual_frame_number}`,
          filename: `${data.video_id_metadata}_${String(data.actual_frame_number).padStart(8, '0')}.webp`,
          video_id: data.video_id_metadata,
          frame_number: data.actual_frame_number,
          image_url: actualLink.image_url ?? `/api/frames/${data.video_id_metadata}/${String(data.actual_frame_number).padStart(8, '0')}.webp`,
          title: `YouTube @ ${data.actual_seconds}s (actual)`,
          fps: data.fps,
          timestamp: String(data.actual_seconds),
          youtube_url: actualLink.youtube_url ?? '',
        },
        {
          id: `yt-${data.video_id}-${data.requested_frame_number}`,
          filename: `${data.video_id_metadata}_${String(data.requested_frame_number).padStart(8, '0')}.webp`,
          video_id: data.video_id_metadata,
          frame_number: data.requested_frame_number,
          image_url: requestedLink.image_url ?? `/api/frames/${data.video_id_metadata}/${String(data.requested_frame_number).padStart(8, '0')}.webp`,
          title: `YouTube @ ${data.requested_seconds}s (requested)`,
          fps: data.fps,
          timestamp: String(data.requested_seconds),
          youtube_url: requestedLink.youtube_url ?? '',
        }
      ];
      console.log(framesRaw)
      const normalized = normalizeToFrameData(framesRaw);
      setFrames(normalized);
      setAllFrames(normalized);
      setSearchStatus(`Found actual frame at ${data.actual_seconds}s, requested frame at ${data.requested_seconds}s`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to search frame by YouTube');
      setSearchStatus('YouTube search failed.');
    } finally {
      setLoading(false);
    }
  }, [youtubeUrl, youtubeSeconds]);


  const searchByVideoID = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching frame by Video ID...');
      const url = `${CONFIG.API_BASE_URL}/api/nearest-frame-link`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          frame_id: Number(frameId),
        }),
      });
      if (!res.ok) throw new Error(`POST /api/nearest-frame-link -> ${res.status}`);
      const data = await res.json();

      // Call /api/youtube-link for actual frame
      const actualLinkRes = await fetch(`${CONFIG.API_BASE_URL}/api/youtube-link?video_id=${videoId}&frame_number=${data.nearest_frame_number}`);
      const actualLink = actualLinkRes.ok ? await actualLinkRes.json() : {};

      // Call /api/youtube-link for requested frame
      const requestedLinkRes = await fetch(`${CONFIG.API_BASE_URL}/api/youtube-link?video_id=${data.video_id_metadata}&frame_number=${data.requested_frame_id}`);
      const requestedLink = requestedLinkRes.ok ? await requestedLinkRes.json() : {};

      const framesRaw = [
        {
          id: `yt-${data.video_id}-${data.actual_frame_number}`,
          filename: `${data.video_id_metadata}_${String(data.actual_frame_number).padStart(8, '0')}.webp`,
          video_id: data.video_id_metadata,
          frame_number: data.actual_frame_number,
          image_url: actualLink.image_url ?? `/api/frames/${data.video_id_metadata}/${String(data.actual_frame_number).padStart(8, '0')}.webp`,
          title: `YouTube @ ${data.actual_seconds}s (nearest)`,
          fps: data.fps,
          timestamp: String(data.actual_seconds),
          youtube_url: actualLink.youtube_url ?? '',
        },
        {
          id: `yt-${data.video_id}-${data.requested_frame_number}`,
          filename: `${data.video_id_metadata}_${String(data.requested_frame_number).padStart(8, '0')}.webp`,
          video_id: data.video_id_metadata,
          frame_number: data.requested_frame_number,
          image_url: requestedLink.image_url ?? `/api/frames/${data.video_id_metadata}/${String(data.requested_frame_number).padStart(8, '0')}.webp`,
          title: `YouTube @ ${data.requested_seconds}s (requested)`,
          fps: data.fps,
          timestamp: String(data.requested_seconds),
          youtube_url: requestedLink.youtube_url ?? '',
        }
      ];
      console.log(framesRaw)
      const normalized = normalizeToFrameData(framesRaw);
      setFrames(normalized);
      setAllFrames(normalized);
      setSearchStatus(`Found actual frame at ${data.actual_seconds}s, requested frame at ${data.requested_seconds}s`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to search frame by YouTube');
      setSearchStatus('YouTube search failed.');
    } finally {
      setLoading(false);
    }
  }, [youtubeUrl, youtubeSeconds]);

  // Audio/ASR → làm giống OCR (POST text, nhận JSON, normalize, setFrames...)
  const searchByAudio = useCallback(async (text: string) => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching Audio/ASR ...');

      const url = `${CONFIG.API_BASE_URL}${API_ENDPOINTS.SEARCH_AUDIO}`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`POST ${API_ENDPOINTS.SEARCH_AUDIO} -> ${res.status}`);

      const payload = await res.json();
      const rawArray: any[] = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.results)
        ? payload.results
        : [];

      const normalized = normalizeToFrameData(rawArray);

      setAllFrames(normalized);
      setFrames(normalized);
      setSearchStatus(`Found ${normalized.length} audio segment(s)`);
      setRetryCount(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Audio search failed';
      setError(msg);
      setSearchStatus('Audio search failed.');
    } finally {
      setLoading(false);
    }
  }, []);

  const searchByOCR = useCallback(async (text: string) => {
    try {
      setLoading(true);
      setError(null);
      setSearchStatus('Searching OCR ...');

      const url = `${CONFIG.API_BASE_URL}${API_ENDPOINTS.SEARCH_OCR}`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`POST ${API_ENDPOINTS.SEARCH_OCR} -> ${res.status}`);

      const payload = await res.json();
      const rawArray: any[] = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.results)
        ? payload.results
        : [];

      const normalized = normalizeToFrameData(rawArray);

      setAllFrames(normalized);
      setFrames(normalized);
      setSearchStatus(`Found ${normalized.length} frame(s) with OCR text`);
      setRetryCount(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'OCR search failed';
      setError(msg);
      setSearchStatus('OCR search failed.');
    } finally {
      setLoading(false);
    }
  }, []);

  // ============== YOUTUBE OPEN (via backend) ==============
  const requestYoutubeUrl = async (video_id: string, frameNumberOrSec: number) => {
    const params = new URLSearchParams({
      video_id,
      frame_number: String(Math.floor(frameNumberOrSec)),
    });
    const url = `${CONFIG.API_BASE_URL}${API_ENDPOINTS.YT_LINK}?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${API_ENDPOINTS.YT_LINK} -> ${res.status}`);
    const data = await res.json();
    if (data?.youtube_url) return data.youtube_url as string;
    throw new Error(data?.error || 'No youtube_url returned');
  };

  const openYouTubeAtTimestamp = async (frame: FrameData) => {
    try {
      const yt = await requestYoutubeUrl(frame.video_id, frame.frame_number);
      window.open(yt, '_blank', 'noopener');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Cannot open YouTube URL for this frame');
    }
  };

  // ============== FILTER ORIENTATION ==============
  const applyOrientationFilter = async (o: Orientation) => {
    setOrientation(o);

    if (o === 'All' || allFrames.length === 0) {
      setFrames(allFrames); // dùng lại toàn bộ frames gốc
      setSearchStatus('Showing all frames');
      return;
    }

    const items = allFrames.map(f => ({ video_id: f.video_id, frame_number: f.frame_number }));
    if (items.length === 0) {
      setFrames([]);
      setSearchStatus('No frames to filter.');
      return;
    }

    try {
      const res = await fetch(`${CONFIG.API_BASE_URL}${API_ENDPOINTS.FILTER_FRAMES}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items, orientation: o }),
      });
      if (!res.ok) throw new Error(`POST ${API_ENDPOINTS.FILTER_FRAMES} -> ${res.status}`);

      const filtered = await res.json();

      const whitelist = new Set(filtered.map((x: any) => `${x.video_id}#${String(x.frame_number)}`));

      const updated = allFrames
        .filter(f => whitelist.has(`${f.video_id}#${String(f.frame_number)}`))
        .map(f => {
          const meta = filtered.find((x: any) =>
            x.video_id === f.video_id && String(x.frame_number) === String(f.frame_number)
          );
          return meta
            ? {
                ...f,
                width: meta.width,
                height: meta.height,
                image_url: meta.image_url || f.image_url,
              }
            : f;
        });

      setFrames(updated); // CHỈ cập nhật frames, KHÔNG chạm vào allFrames
      setSearchStatus(`Applied orientation filter: ${o}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Filter failed');
    }
  };

  // ============== SIDE EFFECTS ==============
  useEffect(() => {
    // Reset orientation khi đổi mode; auto load nếu random
    setOrientation('All');
    if (mode === 'random') fetchFrames();
  }, [mode]);

  // Chỉ gọi filter khi orientation thay đổi
  useEffect(() => {
    if (orientation === 'All') {
      setFrames(allFrames);
      setSearchStatus('Showing all frames');
      return;
    }
    if (allFrames.length === 0) return;
    applyOrientationFilter(orientation);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orientation]);

  // ============== ACTIONS ==============
  const handleSearch = () => {
    if (mode === 'random') {
      fetchFrames();
      return;
    }

    if (mode === 'youtube') {
      if (!youtubeUrl.trim() || youtubeSeconds.trim() === '' || isNaN(Number(youtubeSeconds))) {
        setFrames([]);
        setSearchStatus('Please enter YouTube URL and valid seconds.');
        return;
      }
      return void searchByYouTube();
    }
    if (mode === 'video_frame') {
      if (!videoId.trim() || frameId.trim() === '' || isNaN(Number(frameId) )){
        setFrames([]);
        setSearchStatus('Please enter YouTube URL and valid seconds.');
        return;
      }
      return void searchByVideoID();
    }


    if (!searchText.trim()) {
      setFrames([]);
      setSearchStatus('Enter query to search.');
      return;
    }

    if (mode === 'text') return void searchByText(searchText);
    if (mode === 'ocr') return void searchByOCR(searchText);
    if (mode === 'audio') return void searchByAudio(searchText);
  };

  // ============== GROUPING ==============
  const grouped = useMemo(() => {
    const m = new Map<string, FrameData[]>();
    for (const f of frames) {
      if (!m.has(f.video_id)) m.set(f.video_id, []);
      m.get(f.video_id)!.push(f);
    }
    return Array.from(m.entries()); // [ [video_id, FrameData[]], ... ]
  }, [frames]);

  // ============== RENDER ==============
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-red-600 rounded flex items-center justify-center">
                <span className="text-white font-bold text-sm">▶</span>
              </div>
              <span className="text-xl font-semibold text-gray-900">Frame Search</span>
            </div>

            <div className="flex-1 max-w-5xl mx-8 grid grid-cols-12 gap-3 items-center">
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as SearchMode)}
                className="col-span-3 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                title="Search mode"
              >
                <option value="random">Random (/api/frames)</option>
                <option value="text">Text (CLIP)</option>
                <option value="ocr">OCR</option>
                <option value="audio">Audio</option>
                <option value="youtube">YouTube (URL + seconds)</option>
                <option value="video_frame">Video Frame</option>
              </select>

              {mode === 'youtube' ? (
                <>
                  <input
                    type="text"
                    value={youtubeUrl}
                    onChange={e => setYoutubeUrl(e.target.value)}
                    placeholder="YouTube URL"
                    className="col-span-5 px-3 py-2 border rounded-md text-sm"
                  />
                  <input
                    type="number"
                    value={youtubeSeconds}
                    onChange={e => setYoutubeSeconds(e.target.value)}
                    placeholder="Seconds"
                    className="col-span-2 px-3 py-2 border rounded-md text-sm"
                  />
                </>
              ) : mode === 'video_frame' ? (
                  <>
                    <input
                      type="text"
                      value={videoId}
                      onChange={e => setVideoId(e.target.value)}
                      placeholder="Video ID (metadata)"
                      className="col-span-5 px-3 py-2 border rounded-md text-sm"
                    />
                    <input
                      type="number"
                      value={frameId}
                      onChange={e => setFrameId(e.target.value)}
                      placeholder="Frame number"
                      className="col-span-2 px-3 py-2 border rounded-md text-sm"
                    />
                  </>
                ): (
                <div className="col-span-7 relative">
                  <input
                    type="text"
                    value={searchText}
                    onChange={e => setSearchText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}
                    disabled={mode === 'random'}
                    className="w-full px-4 py-2 pl-10 pr-12 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                    placeholder={
                      mode === 'random'
                        ? 'Switch to a mode to search...'
                        : (mode === 'ocr' ? 'Search text inside frames (OCR)...'
                          : mode === 'audio' ? 'Search in transcript...'
                          : 'Search frames by text...')
                    }
                  />
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                  </div>
                  <button
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => {
                      setSearchText('');
                      if (mode !== 'random') {
                        setFrames([]);
                        setSearchStatus('Enter query to search.');
                      }
                    }}
                    aria-label="Clear"
                    title="Clear search"
                    type="button"
                  >
                    <svg className="h-5 w-5 text-gray-400 hover:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                  </button>
                </div>
              )}

              <button
                onClick={handleSearch}
                className="col-span-1 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
                title="Search"
              >
                Search
              </button>

              <select
                value={orientation}
                onChange={(e) => applyOrientationFilter(e.target.value as Orientation)}
                className="col-span-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                title="Filter orientation"
                disabled={frames.length === 0}
              >
                <option value="All">Orientation: All</option>
                <option value="Ngang">Ngang</option>
                <option value="Dọc">Dọc</option>
                <option value="Khác">Khác</option>
              </select>
            </div>

            <div className="text-xs text-gray-400">{searchStatus}</div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-4">
          {loading && <p className="text-sm text-gray-600">Loading...</p>}
          {error && <p className="text-sm text-red-600">Error: {error}</p>}
          {!loading && !error && frames.length > 0 && (
            <p className="text-sm text-gray-600">
              Showing <span className="font-medium">{frames.length}</span> items
            </p>
          )}
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <p className="text-red-600 mb-4">Failed: {error}</p>
            {mode === 'random' && (
              <button
                onClick={() => fetchFrames()}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Retry
              </button>
            )}
          </div>
        ) : frames.length === 0 ? (
          <div className="text-center text-sm text-gray-500 py-16">
            {mode === 'random' ? 'No frames.' : 'No results. Try again.'}
          </div>
        ) : (
          <div className="space-y-8">
            {grouped.map(([videoId, list], idx) => (
              <section key={videoId}>
                {idx > 0 && <hr className="my-4 border-gray-200" />}
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-lg font-semibold text-gray-800">
                    🎞️ Video: {videoId}
                  </h2>
                  <button
                    onClick={() => openYouTubeAtTimestamp(list[0])}
                    className="text-xs px-2 py-1 border rounded hover:bg-gray-50"
                    title="Open video at first item"
                  >
                    Open on YouTube
                  </button>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                  {list.map((frame, frameIdx) => {
                    // For youtube mode, show which is actual/requested
                    let youtubeLabel = '';
                    if (mode === 'youtube') {
                      if (frameIdx === 0) youtubeLabel = 'Actual Frame';
                      if (frameIdx === 1) youtubeLabel = 'Requested Frame';
                    }
                    return (
                      <div
                        key={frame.id}
                        className="bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200 cursor-pointer"
                        onClick={() => openYouTubeAtTimestamp(frame)}
                        title="Open on YouTube at this exact moment"
                        role="button"
                      >
                        <div className="aspect-video bg-gray-300 rounded-t-lg relative overflow-hidden">
                          <img
                            src={`${CONFIG.API_BASE_URL}${frame.image_url}`}
                            alt={frame.filename}
                            className="w-full h-full object-cover"
                            loading="lazy"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.style.display = 'none';
                              const fallback = target.parentElement?.querySelector('.fallback-icon') as HTMLElement;
                              if (fallback) fallback.setAttribute('style', 'display:flex');
                            }}
                          />
                          <div className="fallback-icon absolute inset-0 bg-gradient-to-br from-gray-200 to-gray-400 items-center justify-center hidden">
                            <svg className="w-12 h-12 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2z" />
                            </svg>
                          </div>
                          <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-1 py-0.5 rounded">
                            #{frame.frame_number}
                          </div>
                          {mode === 'youtube' && youtubeLabel && (
                            <div className="absolute top-2 left-2 bg-blue-600 bg-opacity-80 text-white text-xs px-2 py-0.5 rounded shadow">
                              {youtubeLabel}
                            </div>
                          )}
                        </div>
                        <div className="p-3">
                          <h3 className="text-sm font-medium text-gray-900 truncate mb-1" title={frame.filename}>
                            {frame.filename}
                          </h3>
                          <p className="text-xs text-gray-400 mt-1">
                            Frame: {frame.frame_number}
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        )}

        {/* Random reload */}
        {mode === 'random' && !loading && !error && frames.length > 0 && (
          <div className="mt-8 text-center">
            <button
              onClick={() => fetchFrames()}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-200"
            >
              Load New Random Frames
            </button>
          </div>
        )}
      </main>
    </div>
  );
};

export default SearchResults;