#!/usr/bin/env python3
"""
Script để chạy FastAPI server
"""
import uvicorn
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Frame Video API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload (development mode)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (production mode)')

    args = parser.parse_args()

    # Thêm thư mục hiện tại vào Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    mode = "Development" if args.reload else "Production"
    print(f"🚀 Starting Frame Video API Server ({mode} mode)...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API docs at: http://{args.host}:{args.port}/docs")

    if args.reload:
        print("🔄 Auto-reload enabled for development")
    else:
        print(f"⚡ Running with {args.workers} worker(s) for production")

    print("-" * 50)

    try:
        if args.reload:
            # Development mode
            uvicorn.run(
                "main:app",
                host=args.host,
                port=args.port,
                reload=True,
                log_level="info",
                access_log=False
            )
        else:
            # Production mode
            uvicorn.run(
                "main:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level="info",
                access_log=True
            )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()