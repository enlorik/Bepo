# Bepo

A local Python FastAPI application for storing and searching memories with images, text notes, and GPS coordinates using CLIP embeddings.

## Features

- **Store Memories**: Upload photos with optional text notes and GPS coordinates
- **Semantic Search**: Search memories using natural language queries
- **CLIP Embeddings**: Uses OpenAI's CLIP model for image and text embeddings
- **Cosine Similarity**: Finds the best matching memory based on semantic similarity
- **SQLite Storage**: Local database with efficient BLOB storage for embeddings
- **Image Storage**: Saves uploaded images to local filesystem

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

**Example using curl:**
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

### POST /search

Search memories using a text query. Returns the top matching memory.

**Parameters:**
- `query` (string, required): Search query text

**Example using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -F "query=sunset"
```

**Response:**
```json
{
  "status": "success",
  "match": {
    "id": 1,
    "timestamp": "2024-01-01T12:00:00.000000",
    "image_path": "images/20240101_120000_000000.jpg",
    "note": "Beautiful sunset at the beach",
    "lat": 34.0522,
    "lon": -118.2437,
    "score": 0.87
  }
}
```

### GET /

Returns API information and available endpoints.

## Database Schema

The application uses SQLite with the following schema:

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
- All data stored locally in SQLite database
- Images saved to local filesystem
- All processing done on your machine