# Bepo

A local Python FastAPI application for storing and searching memories with images, text notes, and GPS coordinates using CLIP embeddings.

## Features

- **Store Memories**: Upload photos with optional text notes and GPS coordinates
- **List & Retrieve Memories**: Browse all memories or fetch a single one by id
- **Serve Images**: Dedicated endpoint for serving stored image files
- **Semantic Search**: Search memories using natural language queries with configurable result count
- **CLIP Embeddings**: Uses OpenAI's CLIP model for image and text embeddings (falls back to simple features when CLIP is unavailable)
- **Cosine Similarity**: Ranks matches by semantic similarity
- **SQLite Storage**: Local database with efficient BLOB storage for embeddings
- **Image Storage**: Saves uploaded images to local filesystem

> **Note:** `memories.db` and the `images/` directory are local runtime data and are excluded from version control via `.gitignore`.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### POST /memory

Store a new memory with a photo, optional text note, and GPS coordinates.

**Parameters:**
- `photo` (file, required): Image file to upload
- `note` (string, optional): Text note describing the memory
- `lat` (float, optional): Latitude coordinate
- `lon` (float, optional): Longitude coordinate

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/memory" \
  -F "photo=@path/to/image.jpg" \
  -F "note=Beautiful sunset at the beach" \
  -F "lat=34.0522" \
  -F "lon=-118.2437"
```

**Response:**
```json
{
  "status": "success",
  "memory_id": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "image_path": "images/20240101_120000_000000.jpg",
  "note": "Beautiful sunset at the beach",
  "lat": 34.0522,
  "lon": -118.2437
}
```

---

### GET /memories

Return all saved memories, newest first. Embeddings are not included.

**Example:**
```bash
curl "http://127.0.0.1:8000/memories"
```

**Response:**
```json
[
  {
    "id": 1,
    "timestamp": "2024-01-01T12:00:00.000000",
    "note": "Beautiful sunset at the beach",
    "lat": 34.0522,
    "lon": -118.2437,
    "image_path": "images/20240101_120000_000000.jpg",
    "image_url": "/image/1",
    "map_url": "https://www.google.com/maps/search/?api=1&query=34.0522,-118.2437"
  }
]
```

---

### GET /memory/{id}

Return a single memory by id. Returns 404 if not found.

**Example:**
```bash
curl "http://127.0.0.1:8000/memory/1"
```

**Response:**
```json
{
  "id": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "note": "Beautiful sunset at the beach",
  "lat": 34.0522,
  "lon": -118.2437,
  "image_path": "images/20240101_120000_000000.jpg",
  "image_url": "/image/1",
  "map_url": "https://www.google.com/maps/search/?api=1&query=34.0522,-118.2437"
}
```

---

### GET /image/{id}

Serve the image file associated with a memory. Designed for use by a mobile frontend.
Returns 404 if the memory or its image file does not exist.

**Example:**
```bash
curl "http://127.0.0.1:8000/image/1" --output photo.jpg
```

---

### POST /search

Search memories using a text query. Returns the top `top_k` matches ranked by similarity score.

**Parameters:**
- `query` (string, required): Search query text (must not be empty)
- `top_k` (int, optional, default 5, min 1, max 20): Number of results to return

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -F "query=sunset" \
  -F "top_k=3"
```

**Response (with results):**
```json
{
  "status": "success",
  "query": "sunset",
  "count": 1,
  "matches": [
    {
      "id": 1,
      "timestamp": "2024-01-01T12:00:00.000000",
      "image_path": "images/20240101_120000_000000.jpg",
      "image_url": "/image/1",
      "note": "Beautiful sunset at the beach",
      "lat": 34.0522,
      "lon": -118.2437,
      "map_url": "https://www.google.com/maps/search/?api=1&query=34.0522,-118.2437",
      "score": 0.87
    }
  ]
}
```

**Response (empty database):**
```json
{
  "status": "no_results",
  "message": "No memories found in database",
  "matches": []
}
```

---

### GET /

Returns API version information and a list of available endpoints.

## Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

Tests use a temporary in-memory database and never attempt to download the CLIP model.

## Database Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME NOT NULL,
    lat REAL,
    lon REAL,
    image_path TEXT NOT NULL,
    image_emb BLOB NOT NULL,
    text_note TEXT,
    text_emb BLOB
);
```

## Architecture

- **FastAPI**: Web framework for the REST API
- **SQLite**: Local database for storing memories and embeddings
- **CLIP (openai/clip-vit-base-patch32)**: Generates embeddings for images and text
- **PIL (Pillow)**: Image processing
- **PyTorch**: Deep learning framework for CLIP
- **NumPy**: Efficient array operations and cosine similarity calculation

## Local Only

This application runs entirely locally:
- No external API calls (except for initial CLIP model download)
- All data stored locally in SQLite database (`memories.db`)
- Images saved to local `images/` directory
- Both `memories.db` and `images/` are excluded from git
- All processing done on your machine