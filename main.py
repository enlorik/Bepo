import os
import sqlite3
import io
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# Database configuration
DB_PATH = "memories.db"
IMAGES_DIR = "images"

# Try to import and use CLIP if available, otherwise use fallback
USE_CLIP = False
model = None
processor = None

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    import torch.nn.functional as F
    USE_CLIP = True
except ImportError:
    print("CLIP not available, using fallback embeddings")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    init_db()
    init_model()
    yield
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="Bepo - Memory Storage and Search", lifespan=lifespan)

def init_model():
    """Initialize CLIP model for embeddings"""
    global model, processor, USE_CLIP
    
    if not USE_CLIP:
        print("Using fallback embedding method (no CLIP)")
        return
    
    try:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        print("Falling back to simple embeddings")
        USE_CLIP = False

def init_db():
    """Initialize SQLite database and create tables"""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME NOT NULL,
            lat REAL,
            lon REAL,
            image_path TEXT NOT NULL,
            image_emb BLOB NOT NULL,
            text_note TEXT,
            text_emb BLOB
        )
    """)

    # Migration-safe schema updates for existing databases.
    cursor.execute("PRAGMA table_info(memories)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    migration_alter_statements = {
        "user_note": "ALTER TABLE memories ADD COLUMN user_note TEXT",
        "bepo_summary": "ALTER TABLE memories ADD COLUMN bepo_summary TEXT",
        "tags": "ALTER TABLE memories ADD COLUMN tags TEXT",
        "mood": "ALTER TABLE memories ADD COLUMN mood TEXT",
        "place_hint": "ALTER TABLE memories ADD COLUMN place_hint TEXT",
    }
    for column, statement in migration_alter_statements.items():
        if column not in existing_columns:
            cursor.execute(statement)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Generate embedding for an image using CLIP or fallback method"""
    if USE_CLIP and model is not None:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # get_image_features should return a tensor, but defensively handle
        # cases where some transformers versions return a model output object
        if hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            image_features = image_features.last_hidden_state[:, 0, :]
        # Normalize the embedding
        image_features = F.normalize(image_features, dim=-1)
        return image_features.cpu().numpy().flatten()
    else:
        # Fallback: Use simple image features (color histogram + basic stats)
        # Fixed embedding size: 128 dimensions
        # Resize for consistent features
        img_resized = image.resize((64, 64))
        img_array = np.array(img_resized).astype(float)
        
        # Color histogram features (per channel, 10 bins each = 30 features)
        features = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=10, range=(0, 256))
            features.extend(hist)
        
        # Basic statistics per channel (4 stats x 3 channels = 12 features)
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # Spatial features: divide image into 4x4 grid, get average color per cell
        # 16 cells x 3 channels = 48 features
        grid_size = 4
        cell_h = img_resized.height // grid_size
        cell_w = img_resized.width // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = img_array[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :]
                for channel in range(3):
                    features.append(np.mean(cell[:, :, channel]))
        
        # Texture features: edge detection-like (38 features to reach 128 total)
        # Calculate gradient magnitude for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            # Horizontal gradients
            grad_x = np.abs(np.diff(channel_data, axis=1)).mean()
            # Vertical gradients
            grad_y = np.abs(np.diff(channel_data, axis=0)).mean()
            features.extend([grad_x, grad_y])
        
        # Pad remaining space to reach exactly 128 dimensions
        current_len = len(features)
        if current_len < 128:
            features.extend([0.0] * (128 - current_len))
        
        # Normalize
        embedding = np.array(features[:128], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

def get_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using CLIP or fallback method"""
    if USE_CLIP and model is not None:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        # get_text_features should return a tensor, but defensively handle
        # cases where some transformers versions return a model output object
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'last_hidden_state'):
            text_features = text_features.last_hidden_state[:, 0, :]
        # Normalize the embedding
        text_features = F.normalize(text_features, dim=-1)
        return text_features.cpu().numpy().flatten()
    else:
        # Fallback: Simple text embedding - Fixed 128 dimensions
        # Convert to lowercase for consistency
        text_lower = text.lower()
        
        # Character frequency features (a-z = 26, 0-9 = 10, space = 1, other = 1)
        features = np.zeros(128, dtype=np.float32)
        
        # Character frequencies (38 features)
        for char in text_lower:
            if 'a' <= char <= 'z':
                features[ord(char) - ord('a')] += 1
            elif '0' <= char <= '9':
                features[26 + ord(char) - ord('0')] += 1
            elif char == ' ':
                features[36] += 1
            else:
                features[37] += 1  # Other characters
        
        # Text length features (2 features)
        features[38] = len(text)
        features[39] = len(text.split())
        
        # N-gram features: bigram and trigram character patterns (40 features)
        words = text_lower.split()
        for i, word in enumerate(words[:20]):  # First 20 words
            if i < 20 and len(word) > 0:
                features[40 + i*2] = ord(word[0]) / 255.0  # First char of word
                features[40 + i*2 + 1] = len(word) / 20.0  # Word length normalized
        
        # Word position features (remaining features up to 128)
        # These capture some positional information
        for i, word in enumerate(words[:48]):
            if 80 + i < 128:
                features[80 + i] = (i + 1) / len(words) if len(words) > 0 else 0
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features

def get_db_connection() -> sqlite3.Connection:
    """Open a SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_map_url(lat, lon) -> Optional[str]:
    """Return a Google Maps URL when both lat and lon are present, else None."""
    if lat is not None and lon is not None:
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    return None


def build_image_url(memory_id: int) -> str:
    """Return the relative URL used to serve an image for a given memory id."""
    return f"/image/{memory_id}"


def row_to_memory_response(row: sqlite3.Row) -> dict:
    """Convert a database row to the standard memory response dict (no embeddings)."""
    memory_id = row["id"]
    lat = row["lat"]
    lon = row["lon"]
    return {
        "id": memory_id,
        "timestamp": row["ts"],
        "note": row["text_note"],
        "user_note": row["user_note"],
        "bepo_summary": row["bepo_summary"],
        "tags": row["tags"],
        "mood": row["mood"],
        "place_hint": row["place_hint"],
        "lat": lat,
        "lon": lon,
        "image_path": row["image_path"],
        "image_url": build_image_url(memory_id),
        "map_url": build_map_url(lat, lon),
    }


def fetch_memory_by_id(memory_id: int) -> Optional[sqlite3.Row]:
    """Return the database row for *memory_id*, or None if not found."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, lat, lon, image_path, image_emb, text_note, text_emb, "
            "user_note, bepo_summary, tags, mood, place_hint "
            "FROM memories WHERE id = ?",
            (memory_id,),
        )
        return cursor.fetchone()
    finally:
        conn.close()


def build_combined_text(*fields: Optional[str]) -> Optional[str]:
    """Combine non-empty text fields for text embedding generation."""
    parts = [field.strip() for field in fields if field is not None and field.strip()]
    if not parts:
        return None
    return "\n".join(parts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Assumes vectors are already normalized. If not normalized, normalizes them first.
    """
    # Normalize vectors to be defensive
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 > 0:
        vec1 = vec1 / norm1
    if norm2 > 0:
        vec2 = vec2 / norm2

    return float(np.dot(vec1, vec2))

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for storage"""
    buffer = io.BytesIO()
    np.save(buffer, embedding)
    return buffer.getvalue()

def deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array"""
    buffer = io.BytesIO(data)
    return np.load(buffer)

@app.post("/memory")
async def create_memory(
    photo: UploadFile = File(...),
    note: Optional[str] = Form(None),
    user_note: Optional[str] = Form(None),
    bepo_summary: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    mood: Optional[str] = Form(None),
    place_hint: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None)
):
    """
    Store a new memory with photo, optional note, and GPS coordinates.
    Generates embeddings for both image and text.
    """
    try:
        # Read and validate image
        image_data = await photo.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Generate unique filename
        timestamp = datetime.now()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = os.path.join(IMAGES_DIR, filename)
        
        # Save image
        image.save(image_path)
        
        # Generate embeddings
        image_emb = get_image_embedding(image)
        text_emb = None
        combined_text = build_combined_text(note, user_note, bepo_summary, tags, mood, place_hint)
        if combined_text is not None:
            text_emb = get_text_embedding(combined_text)
        
        # Store in database
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (
                    ts, lat, lon, image_path, image_emb, text_note, text_emb,
                    user_note, bepo_summary, tags, mood, place_hint
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                lat,
                lon,
                image_path,
                serialize_embedding(image_emb),
                note,
                serialize_embedding(text_emb) if text_emb is not None else None,
                user_note,
                bepo_summary,
                tags,
                mood,
                place_hint,
            ))
            memory_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "timestamp": timestamp.isoformat(),
            "image_path": image_path,
            "image_url": build_image_url(memory_id),
            "note": note,
            "user_note": user_note,
            "bepo_summary": bepo_summary,
            "tags": tags,
            "mood": mood,
            "place_hint": place_hint,
            "lat": lat,
            "lon": lon,
            "map_url": build_map_url(lat, lon),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating memory: {str(e)}")

@app.post("/search")
async def search_memories(
    query: str = Form(...),
    top_k: int = Form(5),
):
    """
    Search memories by text query.
    Returns up to *top_k* matches sorted by score descending.
    """
    # Validate inputs
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty or whitespace")
    if not (1 <= top_k <= 20):
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")

    try:
        # Generate query embedding
        query_emb = get_text_embedding(query)

        # Retrieve all memories with embeddings
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, ts, lat, lon, image_path, image_emb, text_note, text_emb, "
                "user_note, bepo_summary, tags, mood, place_hint "
                "FROM memories"
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "status": "no_results",
                "message": "No memories found in database",
                "matches": [],
            }

        # Score every memory
        scored = []
        for row in rows:
            image_emb = deserialize_embedding(row["image_emb"])
            image_score = cosine_similarity(query_emb, image_emb)

            text_score = -1.0
            if row["text_emb"] is not None:
                text_emb = deserialize_embedding(row["text_emb"])
                text_score = cosine_similarity(query_emb, text_emb)

            score = max(image_score, text_score)
            memory_id = row["id"]
            lat = row["lat"]
            lon = row["lon"]
            scored.append({
                "id": memory_id,
                "timestamp": row["ts"],
                "image_path": row["image_path"],
                "image_url": build_image_url(memory_id),
                "note": row["text_note"],
                "user_note": row["user_note"],
                "bepo_summary": row["bepo_summary"],
                "tags": row["tags"],
                "mood": row["mood"],
                "place_hint": row["place_hint"],
                "lat": lat,
                "lon": lon,
                "map_url": build_map_url(lat, lon),
                "score": score,
            })

        # Sort by score descending and take top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        matches = scored[:top_k]

        return {
            "status": "success",
            "query": query,
            "count": len(matches),
            "matches": matches,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")


@app.get("/memories")
async def list_memories():
    """Return all saved memories, newest first. Embeddings are not included."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, lat, lon, image_path, text_note, "
            "user_note, bepo_summary, tags, mood, place_hint "
            "FROM memories ORDER BY ts DESC"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [row_to_memory_response(row) for row in rows]


@app.get("/memory/{memory_id}")
async def get_memory(memory_id: int):
    """Return a single memory by id. Returns 404 if not found."""
    row = fetch_memory_by_id(memory_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return row_to_memory_response(row)


@app.get("/image/{memory_id}")
async def get_image(memory_id: int):
    """Serve the image file associated with a memory. Returns 404 if not found."""
    row = fetch_memory_by_id(memory_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    image_path = row["image_path"]
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found for memory {memory_id}")
    return FileResponse(image_path)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": "Bepo",
        "version": "0.3",
        "description": "Memory storage and search with image and text embeddings",
        "endpoints": {
            "POST /memory": "Store a new memory with photo, note, and GPS",
            "GET /memories": "List all memories, newest first",
            "GET /memory/{id}": "Get a single memory by id",
            "GET /image/{id}": "Serve the image for a memory",
            "POST /search": "Search memories by text query (supports top_k parameter)",
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
