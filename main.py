import os
import sqlite3
import io
import base64
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
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
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Generate embedding for an image using CLIP or fallback method"""
    if USE_CLIP and model is not None:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Assumes vectors are already normalized. If not normalized, normalizes them first.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
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
        if note:
            text_emb = get_text_embedding(note)
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (ts, lat, lon, image_path, image_emb, text_note, text_emb)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp.isoformat(),
            lat,
            lon,
            image_path,
            serialize_embedding(image_emb),
            note,
            serialize_embedding(text_emb) if text_emb is not None else None
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "timestamp": timestamp.isoformat(),
            "image_path": image_path,
            "note": note,
            "lat": lat,
            "lon": lon
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating memory: {str(e)}")

@app.post("/search")
async def search_memories(query: str = Form(...)):
    """
    Search memories by text query.
    Returns the top matching memory based on cosine similarity of embeddings.
    """
    try:
        # Generate query embedding
        query_emb = get_text_embedding(query)
        
        # Retrieve all memories with embeddings
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, ts, lat, lon, image_path, image_emb, text_note, text_emb
            FROM memories
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "status": "no_results",
                "message": "No memories found in database"
            }
        
        # Calculate similarities
        best_match = None
        best_score = -1
        
        for row in rows:
            memory_id, ts, lat, lon, image_path, image_emb_blob, text_note, text_emb_blob = row
            
            # Deserialize embeddings
            image_emb = deserialize_embedding(image_emb_blob)
            
            # Calculate similarity with image embedding
            image_score = cosine_similarity(query_emb, image_emb)
            
            # If text embedding exists, also calculate text similarity and use the better one
            text_score = -1
            if text_emb_blob:
                text_emb = deserialize_embedding(text_emb_blob)
                text_score = cosine_similarity(query_emb, text_emb)
            
            # Use the maximum similarity score
            score = max(image_score, text_score)
            
            if score > best_score:
                best_score = score
                best_match = {
                    "id": memory_id,
                    "timestamp": ts,
                    "image_path": image_path,
                    "note": text_note,
                    "lat": lat,
                    "lon": lon,
                    "score": score
                }
        
        return {
            "status": "success",
            "match": best_match
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": "Bepo",
        "description": "Memory storage and search with image and text embeddings",
        "endpoints": {
            "/memory": "POST - Store a new memory with photo, note, and GPS",
            "/search": "POST - Search memories by text query"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
