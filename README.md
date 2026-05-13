# Bepo

A local Python FastAPI application for storing and searching memories with images, text notes, richer memory metadata, and GPS coordinates using CLIP embeddings.

## Features

- **Store Memories**: Upload photos with optional notes, richer metadata, and GPS coordinates
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

Store a new memory with a photo, optional text note/metadata, and GPS coordinates.

**Parameters:**
- `photo` (file, required): Image file to upload
- `note` (string, optional): Text note describing the memory
- `user_note` (string, optional): What the user noticed/felt in their own words
- `bepo_summary` (string, optional): Optional concise assistant-style summary of the memory
- `tags` (string, optional): Comma-separated or free-form tags
- `mood` (string, optional): Mood/emotion hint (for example `calm`, `excited`)
- `place_hint` (string, optional): Optional place cue (for example `near red couch hallway`)
- `lat` (float, optional): Latitude coordinate
- `lon` (float, optional): Longitude coordinate

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/memory" \
  -F "photo=@path/to/image.jpg" \
  -F "note=Beautiful sunset at the beach" \
  -F "mood=calm" \
  -F "tags=beach,sunset" \
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
  "image_url": "/image/1",
  "note": "Beautiful sunset at the beach",
  "user_note": null,
  "bepo_summary": null,
  "tags": "beach,sunset",
  "mood": "calm",
  "place_hint": null,
  "lat": 34.0522,
  "lon": -118.2437,
  "map_url": "https://www.google.com/maps/search/?api=1&query=34.0522,-118.2437"
}
```

---

### PATCH /memory/{id}/metadata

Update editable memory metadata after a memory is created. This prepares Bepo for future chat-based memory annotation by letting later context refine what Bepo remembers about a photo.

**Request body (JSON, all fields optional):**
- `note` (string or `null`)
- `user_note` (string or `null`)
- `bepo_summary` (string or `null`)
- `tags` (string or `null`)
- `mood` (string or `null`)
- `place_hint` (string or `null`)

If a field is omitted, the existing value is preserved. If a field is sent as `null`, that field is cleared. After every metadata update, Bepo rebuilds the text embedding from the current text fields so search reflects the updated memory meaning.

**Example:**
```bash
curl -X PATCH "http://127.0.0.1:8000/memory/1/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "user_note": "I remember the quiet corner table",
    "tags": "cafe,quiet,corner",
    "mood": "peaceful",
    "place_hint": "near the back window"
  }'
```

**Response:**
```json
{
  "id": 1,
  "timestamp": "2024-01-01T12:00:00.000000",
  "note": "Beautiful sunset at the beach",
  "user_note": "I remember the quiet corner table",
  "bepo_summary": null,
  "tags": "cafe,quiet,corner",
  "mood": "peaceful",
  "place_hint": "near the back window",
  "lat": 34.0522,
  "lon": -118.2437,
  "image_path": "images/20240101_120000_000000.jpg",
  "image_url": "/image/1",
  "map_url": "https://www.google.com/maps/search/?api=1&query=34.0522,-118.2437"
}
```

---

### GET /memories

Return all saved memories, newest first. Embeddings are not included.
Includes `note`, `user_note`, `bepo_summary`, `tags`, `mood`, `place_hint`, `lat`, `lon`, `map_url`, and `image_url`.

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
Includes `note`, `user_note`, `bepo_summary`, `tags`, `mood`, `place_hint`, `lat`, `lon`, `map_url`, and `image_url`.

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
      "user_note": null,
      "bepo_summary": null,
      "tags": "beach,sunset",
      "mood": "calm",
      "place_hint": null,
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
    text_emb BLOB,
    user_note TEXT,
    bepo_summary TEXT,
    tags TEXT,
    mood TEXT,
    place_hint TEXT
);
```

`text_note` is kept for backward compatibility. On startup, Bepo runs a migration-safe schema check (`PRAGMA table_info(memories)` + `ALTER TABLE ... ADD COLUMN`) so existing `memories.db` files are upgraded in place without dropping data.

For semantic search, Bepo builds text embeddings from all available memory text fields (`note`, `user_note`, `bepo_summary`, `tags`, `mood`, `place_hint`). These richer fields are intended for future Bepo chat and personal recall features.

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
