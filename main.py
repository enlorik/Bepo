import os
import sqlite3
import io
import base64
from datetime import datetime
from typing import Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Bepo - Memory Storage and Search")

# Database configuration
DB_PATH = "memories.db"
IMAGES_DIR = "images"

# Global variables for CLIP model
model = None
processor = None

def init_model():
    """Initialize CLIP model for embeddings"""
    global model, processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully")

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
    """Generate embedding for an image using CLIP"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the embedding
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

def get_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using CLIP"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize the embedding
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
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

@app.on_event("startup")
async def startup_event():
    """Initialize database and model on startup"""
    init_db()
    init_model()

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
