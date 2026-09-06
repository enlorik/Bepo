import os
import re
import secrets
import sqlite3
import io
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Optional
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, FastAPI, UploadFile, File, Form, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Runtime configuration. Railway provides RAILWAY_VOLUME_MOUNT_PATH when a
# persistent volume is attached, while local development continues to use the
# repository directory by default.
DATA_DIR = (
    os.getenv("BEPO_DATA_DIR")
    or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    or "."
)
DB_PATH = os.getenv("BEPO_DB_PATH") or os.path.join(DATA_DIR, "memories.db")
IMAGES_DIR = os.getenv("BEPO_IMAGES_DIR") or os.path.join(DATA_DIR, "images")
API_KEY = os.getenv("BEPO_API_KEY") or None

# Keep the downloaded visual model on Bepo's persistent Railway volume. The
# first CLIP startup downloads the model; later deploys can reuse that cache.
MODEL_CACHE_DIR = os.getenv("BEPO_MODEL_CACHE_DIR") or os.path.join(DATA_DIR, "model-cache")
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag with conservative parsing."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# CLIP is opt-in so small hosted deployments can start quickly and reliably.
# Install requirements-clip.txt and set BEPO_ENABLE_CLIP=1 to enable it.
USE_CLIP = env_flag("BEPO_ENABLE_CLIP")
CLIP_MIN_SIMILARITY = float(os.getenv("BEPO_CLIP_MIN_SIMILARITY", "0.24"))
CLIP_CLUSTER_MAX_GAP = float(os.getenv("BEPO_CLIP_CLUSTER_MAX_GAP", "0.025"))
CLIP_CLUSTER_MAX_SPREAD = float(os.getenv("BEPO_CLIP_CLUSTER_MAX_SPREAD", "0.05"))
model = None
processor = None

if USE_CLIP:
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("CLIP dependencies are unavailable; using fallback embeddings")
        USE_CLIP = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    init_db()
    init_model()
    refresh_embeddings_for_active_model()
    yield
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="Bepo - Memory Storage and Search", lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(provided_api_key: Optional[str] = Security(api_key_header)):
    """Require the configured API key while keeping local development simple."""
    if API_KEY is None:
        return
    if provided_api_key is None or not secrets.compare_digest(provided_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


router = APIRouter(dependencies=[Depends(require_api_key)])

METADATA_FIELDS = (
    "note",
    "user_note",
    "bepo_summary",
    "tags",
    "mood",
    "place_hint",
    "context_type",
    "shopping_status",
    "place_id",
)

CONTEXT_TYPES = {"physical", "online", "mixed", "unknown"}
SHOPPING_STATUSES = {"want", "ordered", "bought", "returned", "no_longer_want"}


class MemoryMetadataUpdate(BaseModel):
    note: Optional[str] = None
    user_note: Optional[str] = None
    bepo_summary: Optional[str] = None
    tags: Optional[str] = None
    mood: Optional[str] = None
    place_hint: Optional[str] = None
    context_type: Optional[str] = None
    shopping_status: Optional[str] = None
    place_id: Optional[int] = Field(default=None, ge=1)

    @field_validator("context_type")
    @classmethod
    def valid_context_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if cleaned not in CONTEXT_TYPES:
            raise ValueError("context_type must be physical, online, mixed, or unknown")
        return cleaned

    @field_validator("shopping_status")
    @classmethod
    def valid_shopping_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip().lower()
        if cleaned not in SHOPPING_STATUSES:
            raise ValueError("shopping_status must be want, ordered, bought, returned, or no_longer_want")
        return cleaned


class PlaceCreate(BaseModel):
    name: str
    parent_id: Optional[int] = Field(default=None, ge=1)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned


class PlaceUpdate(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[int] = Field(default=None, ge=1)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)

    @field_validator("name")
    @classmethod
    def optional_name_not_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned

class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=3, ge=1, le=10)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)
    place_id: Optional[int] = Field(default=None, ge=1)
    detect_places: bool = True

    @field_validator("message")
    @classmethod
    def message_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty or whitespace")
        return v

def init_model():
    """Initialize CLIP model for embeddings"""
    global model, processor, USE_CLIP
    
    if not USE_CLIP:
        print("Using fallback embedding method (no CLIP)")
        return
    
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
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
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS places (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            parent_id INTEGER,
            lat REAL,
            lon REAL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            FOREIGN KEY(parent_id) REFERENCES places(id) ON DELETE RESTRICT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME NOT NULL,
            taken_at DATETIME,
            taken_at_source TEXT,
            lat REAL,
            lon REAL,
            location_source TEXT,
            image_path TEXT NOT NULL,
            image_emb BLOB NOT NULL,
            text_note TEXT,
            text_emb BLOB,
            note_created_at DATETIME,
            note_updated_at DATETIME,
            context_type TEXT NOT NULL DEFAULT 'unknown',
            shopping_status TEXT,
            shopping_status_updated_at DATETIME,
            place_id INTEGER,
            FOREIGN KEY(place_id) REFERENCES places(id) ON DELETE SET NULL
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
        "taken_at": "ALTER TABLE memories ADD COLUMN taken_at DATETIME",
        "taken_at_source": "ALTER TABLE memories ADD COLUMN taken_at_source TEXT",
        "location_source": "ALTER TABLE memories ADD COLUMN location_source TEXT",
        "note_created_at": "ALTER TABLE memories ADD COLUMN note_created_at DATETIME",
        "note_updated_at": "ALTER TABLE memories ADD COLUMN note_updated_at DATETIME",
        "context_type": "ALTER TABLE memories ADD COLUMN context_type TEXT NOT NULL DEFAULT 'unknown'",
        "shopping_status": "ALTER TABLE memories ADD COLUMN shopping_status TEXT",
        "shopping_status_updated_at": "ALTER TABLE memories ADD COLUMN shopping_status_updated_at DATETIME",
        "place_id": "ALTER TABLE memories ADD COLUMN place_id INTEGER REFERENCES places(id) ON DELETE SET NULL",
    }
    for column, statement in migration_alter_statements.items():
        if column not in existing_columns:
            cursor.execute(statement)

    # Older memories with coordinates are physical by default. Memories without
    # coordinates remain unknown until their owner labels them.
    if "context_type" not in existing_columns:
        cursor.execute("""
            UPDATE memories
            SET context_type = 'physical'
            WHERE lat IS NOT NULL AND lon IS NOT NULL
        """)
    cursor.execute("UPDATE memories SET context_type = 'unknown' WHERE context_type IS NULL")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_places_parent_id ON places(parent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_place_id ON memories(place_id)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            status TEXT,
            changed_at DATETIME NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)

    # Existing notes were written when their memory was first added. Preserve
    # that useful history while leaving the unknown event date untouched.
    cursor.execute("""
        UPDATE memories
        SET note_created_at = ts, note_updated_at = ts
        WHERE note_created_at IS NULL
          AND (TRIM(COALESCE(text_note, '')) <> '' OR TRIM(COALESCE(user_note, '')) <> '')
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
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def fetch_place_by_id(place_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[sqlite3.Row]:
    """Return a place row, optionally reusing an existing connection."""
    owns_connection = conn is None
    active_conn = conn or get_db_connection()
    try:
        return active_conn.execute(
            "SELECT id, name, parent_id, lat, lon, created_at, updated_at FROM places WHERE id = ?",
            (place_id,),
        ).fetchone()
    finally:
        if owns_connection:
            active_conn.close()


def place_path_rows(conn: sqlite3.Connection, place_id: int) -> list[sqlite3.Row]:
    """Return a place's root-to-leaf path while defending against corrupt cycles."""
    path = []
    seen = set()
    current_id: Optional[int] = place_id
    while current_id is not None and current_id not in seen:
        seen.add(current_id)
        row = fetch_place_by_id(current_id, conn)
        if row is None:
            break
        path.append(row)
        current_id = row["parent_id"]
    path.reverse()
    return path


def place_descendant_ids(conn: sqlite3.Connection, place_id: int) -> list[int]:
    """Return a place id and every nested child id."""
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(id) AS (
            SELECT id FROM places WHERE id = ?
            UNION ALL
            SELECT places.id FROM places JOIN descendants ON places.parent_id = descendants.id
        )
        SELECT id FROM descendants
        """,
        (place_id,),
    ).fetchall()
    return [row["id"] for row in rows]


def place_to_response(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """Build a place response with hierarchy and inherited pin information."""
    path_rows = place_path_rows(conn, row["id"])
    effective_row = next(
        (item for item in reversed(path_rows) if item["lat"] is not None and item["lon"] is not None),
        None,
    )
    descendant_ids = place_descendant_ids(conn, row["id"])
    placeholders = ",".join("?" for _ in descendant_ids)
    descendant_memory_count = conn.execute(
        f"SELECT COUNT(*) FROM memories WHERE place_id IN ({placeholders})",
        descendant_ids,
    ).fetchone()[0]
    direct_memory_count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE place_id = ?", (row["id"],)
    ).fetchone()[0]
    child_count = conn.execute(
        "SELECT COUNT(*) FROM places WHERE parent_id = ?", (row["id"],)
    ).fetchone()[0]
    return {
        "id": row["id"],
        "name": row["name"],
        "parent_id": row["parent_id"],
        "lat": row["lat"],
        "lon": row["lon"],
        "effective_lat": effective_row["lat"] if effective_row else None,
        "effective_lon": effective_row["lon"] if effective_row else None,
        "pin_inherited": effective_row is not None and effective_row["id"] != row["id"],
        "path": [{"id": item["id"], "name": item["name"]} for item in path_rows],
        "path_label": " › ".join(item["name"] for item in path_rows),
        "direct_memory_count": direct_memory_count,
        "memory_count": descendant_memory_count,
        "child_count": child_count,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def place_to_search_record(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    """Return the small, stable place shape used by conversational search."""
    path_rows = place_path_rows(conn, row["id"])
    return {
        "id": row["id"],
        "name": row["name"],
        "path": [{"id": item["id"], "name": item["name"]} for item in path_rows],
        "path_label": " › ".join(item["name"] for item in path_rows),
    }


def list_place_search_records(conn: sqlite3.Connection) -> list[dict]:
    """Return every manual place without the heavier counts used by /places."""
    rows = conn.execute(
        "SELECT id, name, parent_id, lat, lon, created_at, updated_at FROM places"
    ).fetchall()
    return [place_to_search_record(conn, row) for row in rows]


def validate_place_parent(
    conn: sqlite3.Connection,
    parent_id: Optional[int],
    place_id: Optional[int] = None,
) -> None:
    """Require an existing parent and prevent a place hierarchy cycle."""
    if parent_id is None:
        return
    if fetch_place_by_id(parent_id, conn) is None:
        raise HTTPException(status_code=422, detail=f"Parent place {parent_id} not found")
    current_id: Optional[int] = parent_id
    seen = set()
    while current_id is not None and current_id not in seen:
        if place_id is not None and current_id == place_id:
            raise HTTPException(status_code=422, detail="A place cannot be inside itself or one of its children")
        seen.add(current_id)
        current = fetch_place_by_id(current_id, conn)
        current_id = current["parent_id"] if current else None


def ensure_unique_place_name(
    conn: sqlite3.Connection,
    name: str,
    parent_id: Optional[int],
    place_id: Optional[int] = None,
) -> None:
    """Prevent confusing duplicate place names under the same parent."""
    duplicate = conn.execute(
        """
        SELECT id FROM places
        WHERE LOWER(name) = LOWER(?)
          AND ((parent_id IS NULL AND ? IS NULL) OR parent_id = ?)
          AND (? IS NULL OR id <> ?)
        """,
        (name, parent_id, parent_id, place_id, place_id),
    ).fetchone()
    if duplicate is not None:
        raise HTTPException(status_code=409, detail="A place with that name already exists here")


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
        "added_at": row["ts"],
        "taken_at": row["taken_at"],
        "taken_at_source": row["taken_at_source"],
        "note_created_at": row["note_created_at"],
        "note_updated_at": row["note_updated_at"],
        "note": row["text_note"],
        "user_note": row["user_note"],
        "bepo_summary": row["bepo_summary"],
        "tags": row["tags"],
        "mood": row["mood"],
        "place_hint": row["place_hint"],
        "context_type": row["context_type"],
        "shopping_status": row["shopping_status"],
        "shopping_status_updated_at": row["shopping_status_updated_at"],
        "place_id": row["place_id"],
        "lat": lat,
        "lon": lon,
        "location_source": row["location_source"],
        "image_path": row["image_path"],
        "image_url": build_image_url(memory_id),
        "map_url": build_map_url(lat, lon),
    }


def metadata_payload_to_updates(payload: MemoryMetadataUpdate) -> dict:
    """Return explicitly set editable fields, mapping API `note` to DB `text_note`."""
    dump_method = getattr(payload, "model_dump", None)
    if dump_method is None:
        provided_fields = payload.dict(exclude_unset=True).keys()
    else:
        provided_fields = dump_method(exclude_unset=True).keys()
    return {
        "text_note" if field == "note" else field: getattr(payload, field)
        for field in METADATA_FIELDS
        if field in provided_fields
    }


def fetch_memory_by_id(memory_id: int) -> Optional[sqlite3.Row]:
    """Return the database row for *memory_id*, or None if not found."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, "
            "image_path, image_emb, text_note, text_emb, note_created_at, note_updated_at, "
            "user_note, bepo_summary, tags, mood, place_hint, context_type, shopping_status, "
            "shopping_status_updated_at, place_id "
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


def active_embedding_size() -> int:
    """Return the vector size produced by the currently active model."""
    if USE_CLIP and model is not None:
        return int(getattr(model.config, "projection_dim", 512))
    return 128


def stored_embedding_size(value: Optional[bytes]) -> Optional[int]:
    """Return a stored vector's size, treating unreadable data as incompatible."""
    if value is None:
        return None
    try:
        return int(deserialize_embedding(value).reshape(-1).size)
    except Exception:
        return None


def refresh_embeddings_for_active_model() -> dict:
    """Upgrade saved memories when Bepo switches embedding models.

    Railway originally used 128-dimensional lightweight vectors while CLIP
    uses 512-dimensional vectors. Rebuilding incompatible rows at startup lets
    existing photos gain visual search without asking the user to re-upload.
    """
    expected_size = active_embedding_size()
    conn = get_db_connection()
    refreshed_images = 0
    refreshed_text = 0
    skipped_images = 0
    try:
        rows = conn.execute(
            "SELECT id, image_path, image_emb, text_emb, text_note, user_note, "
            "bepo_summary, tags, mood, place_hint, taken_at FROM memories"
        ).fetchall()
        for row in rows:
            updates = {}
            if stored_embedding_size(row["image_emb"]) != expected_size:
                try:
                    with Image.open(row["image_path"]) as stored_image:
                        rgb_image = stored_image.convert("RGB")
                        updates["image_emb"] = serialize_embedding(get_image_embedding(rgb_image))
                    refreshed_images += 1
                except Exception as exc:
                    skipped_images += 1
                    print(f"Could not refresh image embedding for memory {row['id']}: {exc}")

            combined_text = build_combined_text(
                row["text_note"],
                row["user_note"],
                row["bepo_summary"],
                row["tags"],
                row["mood"],
                row["place_hint"],
                datetime_search_text(row["taken_at"]),
            )
            if combined_text is not None and stored_embedding_size(row["text_emb"]) != expected_size:
                updates["text_emb"] = serialize_embedding(get_text_embedding(combined_text))
                refreshed_text += 1

            if updates:
                assignments = ", ".join(f"{field} = ?" for field in updates)
                conn.execute(
                    f"UPDATE memories SET {assignments} WHERE id = ?",
                    (*updates.values(), row["id"]),
                )
        conn.commit()
    finally:
        conn.close()

    result = {
        "embedding_size": expected_size,
        "images": refreshed_images,
        "text": refreshed_text,
        "skipped_images": skipped_images,
    }
    print(f"Embedding refresh complete: {result}")
    return result


def normalize_datetime_value(value: Optional[str], field_name: str) -> Optional[str]:
    """Validate an optional ISO date/time while preserving its supplied timezone."""
    if value is None or not value.strip():
        return None
    cleaned = value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be an ISO date or date-time") from exc
    return parsed.isoformat()


def datetime_search_text(value: Optional[str]) -> Optional[str]:
    """Turn a stored date into useful searchable year/month/day words."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y %B %d")


def friendly_memory_date(value: str) -> str:
    """Return a compact human date for Bepo's conversational answer."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%B %d, %Y").replace(" 0", " ")


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


def split_metadata_values(value: Optional[str]) -> set[str]:
    """Return normalized comma-separated metadata values."""
    if value is None:
        return set()
    return {part.strip().casefold() for part in value.split(",") if part.strip()}


def remove_query_phrase(value: str, pattern: str) -> str:
    """Remove a recognized filter phrase while preserving the remaining query."""
    return re.sub(pattern, " ", value, flags=re.IGNORECASE)


def normalize_place_term(value: str) -> str:
    """Normalize typed @place terms and saved names for matching."""
    return re.sub(r"\s+", " ", value.replace("-", " ").replace("_", " ")).strip().casefold()


def place_name_pattern(name: str) -> str:
    """Build a word-bounded pattern that tolerates spaces, hyphens, and underscores."""
    words = re.findall(r"\w+", normalize_place_term(name), flags=re.UNICODE)
    if not words:
        return r"(?!)"
    separator = r"[\s_-]+"
    return rf"(?<!\w){separator.join(re.escape(word) for word in words)}(?!\w)"


def resolve_place_query(
    query: str,
    known_places: list[dict],
    selected_place_id: Optional[int] = None,
    detect_implicit_places: bool = True,
) -> tuple[str, Optional[dict], list[dict], Optional[str]]:
    """Resolve an explicit or conversational place reference without using GPS."""
    working = query
    by_id = {place["id"]: place for place in known_places}

    if selected_place_id is not None:
        selected = by_id.get(selected_place_id)
        if selected is None:
            raise HTTPException(status_code=422, detail=f"Place {selected_place_id} not found")
        working = remove_query_phrase(working, place_name_pattern(selected["name"]))
        working = re.sub(r"(?<!\S)@[\w-]+", " ", working, flags=re.UNICODE)
        return working, selected, [], None

    explicit = re.search(r"(?<!\S)@([\w-]+)", working, flags=re.UNICODE)
    if explicit:
        raw_term = explicit.group(1)
        term = normalize_place_term(raw_term)
        matches = [
            place
            for place in known_places
            if normalize_place_term(place["name"]) == term
            or normalize_place_term(place["path_label"].replace("›", " ")) == term
        ]
        working = f"{working[:explicit.start()]} {working[explicit.end():]}"
        if len(matches) == 1:
            return working, matches[0], [], None
        if len(matches) > 1:
            return working, None, sorted(matches, key=lambda item: item["path_label"].casefold()), None
        return working, None, [], term or None

    if not detect_implicit_places:
        return working, None, [], None

    matched_groups: dict[str, dict] = {}
    for place in known_places:
        normalized_name = normalize_place_term(place["name"])
        if not normalized_name or not re.search(place_name_pattern(place["name"]), working, flags=re.IGNORECASE):
            continue
        group = matched_groups.setdefault(
            normalized_name,
            {
                "places": [],
                "specificity": (
                    len(normalized_name.split()),
                    len(normalized_name),
                    len(place["path"]),
                ),
            },
        )
        group["places"].append(place)
        group["specificity"] = max(
            group["specificity"],
            (len(normalized_name.split()), len(normalized_name), len(place["path"])),
        )

    if not matched_groups:
        return working, None, [], None

    most_specific = max(matched_groups.values(), key=lambda group: group["specificity"])
    matches = most_specific["places"]
    working = remove_query_phrase(working, place_name_pattern(matches[0]["name"]))
    if len(matches) == 1:
        return working, matches[0], [], None
    return working, None, sorted(matches, key=lambda item: item["path_label"].casefold()), None


def parse_memory_query(
    query: str,
    known_moods: set[str],
    known_places: Optional[list[dict]] = None,
    selected_place_id: Optional[int] = None,
    detect_implicit_places: bool = True,
) -> dict:
    """Extract exact filters from a conversational memory query."""
    working = query.casefold()
    working, place, place_options, suggested_place_name = resolve_place_query(
        working, known_places or [], selected_place_id, detect_implicit_places
    )
    hashtags = [match.group(1).casefold() for match in re.finditer(r"(?<!\S)#([\w-]+)", working)]
    working = re.sub(r"(?<!\S)#[\w-]+", " ", working)

    status_aliases = {
        "want": "want",
        "wishlist": "want",
        "to-buy": "want",
        "ordered": "ordered",
        "bought": "bought",
        "purchased": "bought",
        "returned": "returned",
        "refunded": "returned",
        "no-longer-want": "no_longer_want",
    }
    context_aliases = {
        "online": "online",
        "digital": "online",
        "physical": "physical",
        "irl": "physical",
        "mixed": "mixed",
    }

    tags = []
    moods = []
    context_type = None
    shopping_status = None
    for hashtag in hashtags:
        if hashtag in status_aliases:
            shopping_status = status_aliases[hashtag]
        elif hashtag in context_aliases:
            context_type = context_aliases[hashtag]
        elif hashtag.replace("-", " ") in known_moods:
            moods.append(hashtag.replace("-", " "))
        else:
            tags.append(hashtag)

    nearby = bool(re.search(r"\b(?:nearby|closest)\b|\bnear\s+me\b", working, flags=re.IGNORECASE))
    working = remove_query_phrase(working, r"\b(?:nearby|closest)\b|\bnear\s+me\b")

    status_patterns = [
        ("no_longer_want", r"\b(?:no\s+longer\s+want|do\s+not\s+want|don't\s+want)\b"),
        ("returned", r"\b(?:returned|refunded)\b"),
        ("bought", r"\b(?:bought|purchased)\b"),
        ("ordered", r"\bordered\b"),
        ("want", r"\b(?:wishlist|to\s+buy)\b"),
    ]
    for status, pattern in status_patterns:
        if re.search(pattern, working, flags=re.IGNORECASE):
            shopping_status = status
            working = remove_query_phrase(working, pattern)
            break
    if shopping_status is None and ("shopping" in tags or not working.strip().replace("?", "")):
        if re.search(r"\bwant\b", working, flags=re.IGNORECASE):
            shopping_status = "want"
            working = remove_query_phrase(working, r"\bwant\b")

    context_patterns = [
        ("mixed", r"\b(?:mixed|both)\b"),
        ("online", r"\b(?:online|digital)\b"),
        ("physical", r"\b(?:in\s+person|physical|irl)\b"),
    ]
    if context_type is None:
        for context, pattern in context_patterns:
            if re.search(pattern, working, flags=re.IGNORECASE):
                context_type = context
                working = remove_query_phrase(working, pattern)
                break

    for mood in sorted(known_moods, key=len, reverse=True):
        pattern = rf"(?<![\w-]){re.escape(mood)}(?![\w-])"
        if re.search(pattern, working, flags=re.IGNORECASE):
            moods.append(mood)
            working = remove_query_phrase(working, pattern)

    semantic_query = re.sub(r"\s+", " ", working).strip(" .,?!")
    if not normalized_search_tokens(semantic_query):
        semantic_query = ""
    return {
        "tags": list(dict.fromkeys(tags)),
        "moods": list(dict.fromkeys(moods)),
        "context_type": context_type,
        "shopping_status": shopping_status,
        "nearby": nearby,
        "place_id": place["id"] if place else None,
        "place": place,
        "place_options": place_options,
        "suggested_place_name": suggested_place_name,
        "semantic_query": semantic_query,
    }


SEARCH_FILLER_WORDS = {
    "a", "an", "and", "are", "can", "could", "did", "do", "for", "from",
    "give", "have", "i", "in", "is", "me", "memory", "memories", "my", "of",
    "on", "photo", "photos", "picture", "pictures", "please", "show", "stuff", "that",
    "the", "these", "things", "those", "to", "was", "were", "where", "with", "you",
}


def normalized_search_tokens(value: Optional[str]) -> set[str]:
    """Extract useful search words with a tiny plural normalization."""
    if not value:
        return set()
    tokens = set()
    for token in re.findall(r"[\w-]+", value.casefold()):
        if token in SEARCH_FILLER_WORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token and token not in SEARCH_FILLER_WORDS:
            tokens.add(token)
    return tokens


def lexical_relevance(query: str, row: sqlite3.Row) -> float:
    """Return direct word overlap so hosted search never dumps unrelated rows."""
    searchable = build_combined_text(
        row["text_note"],
        row["user_note"],
        row["bepo_summary"],
        row["tags"],
        row["mood"],
        row["place_hint"],
        datetime_search_text(row["taken_at"]),
    ) or ""
    query_tokens = normalized_search_tokens(query)
    if not query_tokens:
        return 0.0
    searchable_tokens = normalized_search_tokens(searchable)
    return len(query_tokens & searchable_tokens) / len(query_tokens)


def keep_top_semantic_cluster(scored: list[dict]) -> list[dict]:
    """Keep direct text matches plus the strong semantic cluster nearest the best score.

    CLIP cosine scores are rankings rather than calibrated probabilities. A fixed
    floor rejects weak results, while the gap/spread rules avoid returning every
    photo merely because each one barely cleared that floor.
    """
    semantic_candidates = sorted(
        (
            item
            for item in scored
            if item["_semantic_score"] >= CLIP_MIN_SIMILARITY
        ),
        key=lambda item: item["_semantic_score"],
        reverse=True,
    )

    selected_ids = set()
    if semantic_candidates:
        best_score = semantic_candidates[0]["_semantic_score"]
        previous_score = best_score
        for item in semantic_candidates:
            score = item["_semantic_score"]
            if best_score - score > CLIP_CLUSTER_MAX_SPREAD:
                break
            if selected_ids and previous_score - score > CLIP_CLUSTER_MAX_GAP:
                break
            selected_ids.add(item["id"])
            previous_score = score

    return [
        item
        for item in scored
        if item["_lexical_score"] > 0 or item["id"] in selected_ids
    ]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two coordinates in kilometers."""
    earth_radius_km = 6371.0088
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, (lat1, lon1, lat2, lon2))
    lat_delta = lat2_r - lat1_r
    lon_delta = lon2_r - lon1_r
    value = sin(lat_delta / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(lon_delta / 2) ** 2
    return earth_radius_km * 2 * atan2(sqrt(value), sqrt(max(0.0, 1 - value)))


def search_memory_matches_with_filters(
    query: str,
    top_k: int,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
    selected_place_id: Optional[int] = None,
    detect_implicit_places: bool = True,
) -> tuple[list, dict, bool]:
    """Return matches, parsed filters, and whether a nearby query needs GPS."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, "
            "image_path, image_emb, text_note, text_emb, note_created_at, note_updated_at, "
            "user_note, bepo_summary, tags, mood, place_hint, context_type, shopping_status, "
            "shopping_status_updated_at, place_id "
            "FROM memories"
        )
        rows = cursor.fetchall()
        known_places = list_place_search_records(conn)
        known_moods = {
            mood
            for row in rows
            for mood in split_metadata_values(row["mood"])
        }
        known_moods.update({"calm", "cozy", "happy", "excited", "nostalgic", "safe", "romantic", "sad", "anxious"})
        filters = parse_memory_query(
            query,
            known_moods,
            known_places,
            selected_place_id,
            detect_implicit_places,
        )
        filters["_has_memories"] = bool(rows)
        place_ids = (
            set(place_descendant_ids(conn, filters["place_id"]))
            if filters["place_id"] is not None
            else set()
        )
    finally:
        conn.close()

    if not rows:
        return [], filters, False
    if filters["place_options"] or filters["suggested_place_name"]:
        return [], filters, False
    needs_location = filters["nearby"] and (current_lat is None or current_lon is None)
    if needs_location:
        return [], filters, True

    semantic_query = filters["semantic_query"]
    query_emb = get_text_embedding(semantic_query) if semantic_query else None
    clip_ready = USE_CLIP and model is not None

    scored = []
    for row in rows:
        if place_ids and row["place_id"] not in place_ids:
            continue
        row_tags = split_metadata_values(row["tags"])
        if not set(filters["tags"]).issubset(row_tags):
            continue
        row_moods = split_metadata_values(row["mood"])
        if not set(filters["moods"]).issubset(row_moods):
            continue
        if filters["shopping_status"] and row["shopping_status"] != filters["shopping_status"]:
            continue

        row_context = row["context_type"] or "unknown"
        context_filter = filters["context_type"]
        if context_filter == "online" and row_context not in {"online", "mixed"}:
            continue
        if context_filter == "physical" and row_context not in {"physical", "mixed"}:
            continue
        if context_filter == "mixed" and row_context != "mixed":
            continue
        if filters["nearby"] and (
            row_context not in {"physical", "mixed"}
            or row["lat"] is None
            or row["lon"] is None
        ):
            continue

        image_score = 1.0 if query_emb is None else -1.0
        if query_emb is not None:
            try:
                image_emb = deserialize_embedding(row["image_emb"])
                if image_emb.size == query_emb.size:
                    image_score = cosine_similarity(query_emb, image_emb)
            except Exception:
                image_score = -1.0

        text_score = -1.0
        if query_emb is not None and row["text_emb"] is not None:
            try:
                text_emb_arr = deserialize_embedding(row["text_emb"])
                if text_emb_arr.size == query_emb.size:
                    text_score = cosine_similarity(query_emb, text_emb_arr)
            except Exception:
                text_score = -1.0

        lexical_score = lexical_relevance(semantic_query, row) if semantic_query else 0.0
        semantic_score = max(image_score, text_score)
        score = max(semantic_score, lexical_score)
        memory_id = row["id"]
        lat = row["lat"]
        lon = row["lon"]
        distance_km = None
        if current_lat is not None and current_lon is not None and lat is not None and lon is not None:
            distance_km = round(haversine_distance_km(current_lat, current_lon, lat, lon), 2)
        scored.append({
            "id": memory_id,
            "timestamp": row["ts"],
            "added_at": row["ts"],
            "taken_at": row["taken_at"],
            "taken_at_source": row["taken_at_source"],
            "note_created_at": row["note_created_at"],
            "note_updated_at": row["note_updated_at"],
            "image_path": row["image_path"],
            "image_url": build_image_url(memory_id),
            "note": row["text_note"],
            "user_note": row["user_note"],
            "bepo_summary": row["bepo_summary"],
            "tags": row["tags"],
            "mood": row["mood"],
            "place_hint": row["place_hint"],
            "context_type": row["context_type"],
            "shopping_status": row["shopping_status"],
            "shopping_status_updated_at": row["shopping_status_updated_at"],
            "place_id": row["place_id"],
            "lat": lat,
            "lon": lon,
            "location_source": row["location_source"],
            "map_url": build_map_url(lat, lon),
            "distance_km": distance_km,
            "score": score,
            "_semantic_score": semantic_score,
            "_lexical_score": lexical_score,
        })

    if semantic_query:
        if clip_ready:
            scored = keep_top_semantic_cluster(scored)
        else:
            scored = [item for item in scored if item["_lexical_score"] > 0]

    if filters["nearby"]:
        scored.sort(key=lambda item: (item["distance_km"], -item["score"]))
    elif semantic_query:
        scored.sort(key=lambda item: item["score"], reverse=True)
    else:
        scored.sort(key=lambda item: item["taken_at"] or item["timestamp"], reverse=True)
    matches = scored[:top_k]
    for item in matches:
        item.pop("_lexical_score", None)
        item.pop("_semantic_score", None)
    return matches, filters, False


def search_memory_matches(query: str, top_k: int) -> list:
    """Backward-compatible semantic/structured search helper."""
    matches, _, _ = search_memory_matches_with_filters(query, top_k)
    return matches


def public_memory_filters(filters: dict) -> dict:
    """Return only user-facing filters, omitting internal semantic-query text."""
    public_keys = {"tags", "moods", "context_type", "shopping_status", "nearby", "place"}
    return {
        key: value
        for key, value in filters.items()
        if key in public_keys and value not in (None, [], False)
    }


def build_filtered_chat_answer(filters: dict, count: int, top: Optional[dict] = None) -> str:
    """Describe an exact filtered result set in friendly language."""
    nearby = filters.get("nearby")
    place = filters.get("place")
    noun = "memory" if count == 1 else "memories"
    ordering = " nearby, closest first" if nearby else ""
    location = f" in {place['path_label']}" if place else ""
    answer = f"I found {count} matching {noun}{location}{ordering}."
    if top is not None and count == 1:
        answer = f"{answer} {build_chat_answer(top)}"
    return answer


def build_chat_answer(top: dict) -> str:
    """Build a simple deterministic answer from the top memory match."""
    sentences = []

    if top.get("place_hint"):
        sentences.append(f"You may mean the memory near {top['place_hint']}.")
    else:
        sentences.append("You may mean this memory.")

    description = top.get("user_note") or top.get("bepo_summary") or top.get("note")
    if description:
        sentences.append(f'I recall: "{description}".')

    if top.get("taken_at"):
        sentences.append(f"The photo was taken on {friendly_memory_date(top['taken_at'])}.")

    details = []
    if top.get("mood"):
        details.extend(value.strip() for value in top["mood"].split(",") if value.strip())
    if top.get("tags"):
        details.extend(t.strip() for t in top["tags"].split(",") if t.strip())
    if details:
        sentences.append(f"I remember it as {', '.join(details)}.")

    if top.get("map_url"):
        sentences.append("A map link is available.")

    return " ".join(sentences)

@router.post("/memory")
async def create_memory(
    photo: UploadFile = File(...),
    note: Optional[str] = Form(None),
    user_note: Optional[str] = Form(None),
    bepo_summary: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    mood: Optional[str] = Form(None),
    place_hint: Optional[str] = Form(None),
    context_type: Optional[str] = Form(None),
    shopping_status: Optional[str] = Form(None),
    place_id: Optional[int] = Form(None),
    taken_at: Optional[str] = Form(None),
    taken_at_source: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    location_source: Optional[str] = Form(None),
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
        
        parsed_taken_at = normalize_datetime_value(taken_at, "taken_at")
        if taken_at_source not in {None, "photo", "camera", "manual"}:
            raise HTTPException(status_code=422, detail="taken_at_source must be photo, camera, or manual")
        if location_source not in {None, "photo", "current", "manual"}:
            raise HTTPException(status_code=422, detail="location_source must be photo, current, or manual")
        normalized_context = context_type.strip().lower() if context_type else None
        if normalized_context is not None and normalized_context not in CONTEXT_TYPES:
            raise HTTPException(status_code=422, detail="context_type must be physical, online, mixed, or unknown")
        normalized_status = shopping_status.strip().lower() if shopping_status else None
        if normalized_status is not None and normalized_status not in SHOPPING_STATUSES:
            raise HTTPException(
                status_code=422,
                detail="shopping_status must be want, ordered, bought, returned, or no_longer_want",
            )
        resolved_context = normalized_context or (
            "physical" if lat is not None and lon is not None else "unknown"
        )
        if place_id is not None and fetch_place_by_id(place_id) is None:
            raise HTTPException(status_code=422, detail=f"Place {place_id} not found")

        # The server timestamp records when the memory/note was added. The
        # optional photo timestamp records when the event itself happened.
        timestamp = datetime.now().astimezone()
        timestamp_iso = timestamp.isoformat()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = os.path.join(IMAGES_DIR, filename)
        
        # Save image
        image.save(image_path)
        
        # Generate embeddings
        image_emb = get_image_embedding(image)
        text_emb = None
        combined_text = build_combined_text(
            note, user_note, bepo_summary, tags, mood, place_hint,
            datetime_search_text(parsed_taken_at),
        )
        if combined_text is not None:
            text_emb = get_text_embedding(combined_text)
        note_created_at = timestamp_iso if build_combined_text(note, user_note) is not None else None
        
        # Store in database
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (
                    ts, taken_at, taken_at_source, lat, lon, location_source,
                    image_path, image_emb, text_note, text_emb, note_created_at, note_updated_at,
                    user_note, bepo_summary, tags, mood, place_hint, context_type,
                    shopping_status, shopping_status_updated_at, place_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp_iso,
                parsed_taken_at,
                taken_at_source if parsed_taken_at is not None else None,
                lat,
                lon,
                location_source if lat is not None and lon is not None else None,
                image_path,
                serialize_embedding(image_emb),
                note,
                serialize_embedding(text_emb) if text_emb is not None else None,
                note_created_at,
                note_created_at,
                user_note,
                bepo_summary,
                tags,
                mood,
                place_hint,
                resolved_context,
                normalized_status,
                timestamp_iso if normalized_status is not None else None,
                place_id,
            ))
            memory_id = cursor.lastrowid
            if normalized_status is not None:
                cursor.execute(
                    "INSERT INTO memory_status_history (memory_id, status, changed_at) VALUES (?, ?, ?)",
                    (memory_id, normalized_status, timestamp_iso),
                )
            conn.commit()
        finally:
            conn.close()
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "timestamp": timestamp_iso,
            "added_at": timestamp_iso,
            "taken_at": parsed_taken_at,
            "taken_at_source": taken_at_source if parsed_taken_at is not None else None,
            "note_created_at": note_created_at,
            "note_updated_at": note_created_at,
            "image_path": image_path,
            "image_url": build_image_url(memory_id),
            "note": note,
            "user_note": user_note,
            "bepo_summary": bepo_summary,
            "tags": tags,
            "mood": mood,
            "place_hint": place_hint,
            "context_type": resolved_context,
            "shopping_status": normalized_status,
            "shopping_status_updated_at": timestamp_iso if normalized_status is not None else None,
            "place_id": place_id,
            "lat": lat,
            "lon": lon,
            "location_source": location_source if lat is not None and lon is not None else None,
            "map_url": build_map_url(lat, lon),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating memory: {str(e)}")


@router.patch("/memory/{memory_id}/metadata")
async def update_memory_metadata(memory_id: int, payload: MemoryMetadataUpdate):
    """Update editable memory metadata fields and refresh the text embedding."""
    existing_row = fetch_memory_by_id(memory_id)
    if existing_row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    updates = metadata_payload_to_updates(payload)
    if "place_id" in updates and updates["place_id"] is not None:
        if fetch_place_by_id(updates["place_id"]) is None:
            raise HTTPException(status_code=422, detail=f"Place {updates['place_id']} not found")
    merged_values = {
        "text_note": existing_row["text_note"],
        "user_note": existing_row["user_note"],
        "bepo_summary": existing_row["bepo_summary"],
        "tags": existing_row["tags"],
        "mood": existing_row["mood"],
        "place_hint": existing_row["place_hint"],
        "context_type": existing_row["context_type"],
        "shopping_status": existing_row["shopping_status"],
        "place_id": existing_row["place_id"],
    }
    merged_values.update(updates)
    if merged_values["context_type"] is None:
        merged_values["context_type"] = "unknown"

    note_created_at = existing_row["note_created_at"]
    note_updated_at = existing_row["note_updated_at"]
    if "text_note" in updates or "user_note" in updates:
        note_updated_at = datetime.now().astimezone().isoformat()
        if note_created_at is None and build_combined_text(
            merged_values["text_note"], merged_values["user_note"]
        ) is not None:
            note_created_at = note_updated_at

    shopping_status_changed = (
        "shopping_status" in updates
        and merged_values["shopping_status"] != existing_row["shopping_status"]
    )
    shopping_status_updated_at = existing_row["shopping_status_updated_at"]
    if shopping_status_changed:
        shopping_status_updated_at = datetime.now().astimezone().isoformat()

    combined_text = build_combined_text(
        merged_values["text_note"],
        merged_values["user_note"],
        merged_values["bepo_summary"],
        merged_values["tags"],
        merged_values["mood"],
        merged_values["place_hint"],
        datetime_search_text(existing_row["taken_at"]),
    )
    serialized_text_emb = None
    if combined_text is not None:
        serialized_text_emb = serialize_embedding(get_text_embedding(combined_text))

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE memories
            SET text_note = ?, user_note = ?, bepo_summary = ?, tags = ?, mood = ?, place_hint = ?,
                context_type = ?, shopping_status = ?, shopping_status_updated_at = ?,
                place_id = ?, text_emb = ?, note_created_at = ?, note_updated_at = ?
            WHERE id = ?
            """,
            (
                merged_values["text_note"],
                merged_values["user_note"],
                merged_values["bepo_summary"],
                merged_values["tags"],
                merged_values["mood"],
                merged_values["place_hint"],
                merged_values["context_type"],
                merged_values["shopping_status"],
                shopping_status_updated_at,
                merged_values["place_id"],
                serialized_text_emb,
                note_created_at,
                note_updated_at,
                memory_id,
            ),
        )
        if shopping_status_changed:
            cursor.execute(
                "INSERT INTO memory_status_history (memory_id, status, changed_at) VALUES (?, ?, ?)",
                (memory_id, merged_values["shopping_status"], shopping_status_updated_at),
            )
        conn.commit()
        cursor.execute(
            "SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, image_path, "
            "text_note, note_created_at, note_updated_at, user_note, bepo_summary, tags, mood, place_hint, "
            "context_type, shopping_status, shopping_status_updated_at, place_id "
            "FROM memories WHERE id = ?",
            (memory_id,),
        )
        updated_row = cursor.fetchone()
    finally:
        conn.close()

    return row_to_memory_response(updated_row)


@router.get("/places")
async def list_places():
    """Return every manually defined place with hierarchy and counts."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, parent_id, lat, lon, created_at, updated_at FROM places"
        ).fetchall()
        places = [place_to_response(conn, row) for row in rows]
        places.sort(key=lambda place: place["path_label"].casefold())
        return places
    finally:
        conn.close()


@router.get("/places/suggestions")
async def suggest_places(
    lat: float,
    lon: float,
    radius_m: float = 250,
    limit: int = 8,
):
    """Suggest existing nearby place pins without assigning anything."""
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Invalid latitude or longitude")
    if not (1 <= radius_m <= 50000):
        raise HTTPException(status_code=422, detail="radius_m must be between 1 and 50000")
    if not (1 <= limit <= 25):
        raise HTTPException(status_code=422, detail="limit must be between 1 and 25")

    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, parent_id, lat, lon, created_at, updated_at FROM places"
        ).fetchall()
        suggestions = []
        for row in rows:
            place = place_to_response(conn, row)
            # Only independently pinned places are suggestions. Children that
            # inherit the same pin stay available inside their parent without
            # flooding the nearby list with duplicate coordinates.
            if place["lat"] is None or place["lon"] is None:
                continue
            distance_km = haversine_distance_km(
                lat, lon, place["lat"], place["lon"]
            )
            if distance_km * 1000 <= radius_m:
                suggestions.append({**place, "distance_m": round(distance_km * 1000)})
        suggestions.sort(key=lambda place: (place["distance_m"], place["path_label"].casefold()))
        return suggestions[:limit]
    finally:
        conn.close()


@router.post("/places")
async def create_place(payload: PlaceCreate):
    """Create a manual root place or a place nested inside another place."""
    if (payload.lat is None) != (payload.lon is None):
        raise HTTPException(status_code=422, detail="lat and lon must be provided together")
    timestamp = datetime.now().astimezone().isoformat()
    conn = get_db_connection()
    try:
        validate_place_parent(conn, payload.parent_id)
        ensure_unique_place_name(conn, payload.name, payload.parent_id)
        cursor = conn.execute(
            """
            INSERT INTO places (name, parent_id, lat, lon, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (payload.name, payload.parent_id, payload.lat, payload.lon, timestamp, timestamp),
        )
        place_id = cursor.lastrowid
        conn.commit()
        row = fetch_place_by_id(place_id, conn)
        return place_to_response(conn, row)
    finally:
        conn.close()


@router.patch("/places/{place_id}")
async def update_place(place_id: int, payload: PlaceUpdate):
    """Rename, move, or update the optional pin for an existing place."""
    conn = get_db_connection()
    try:
        existing = fetch_place_by_id(place_id, conn)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Place {place_id} not found")
        dump_method = getattr(payload, "model_dump", None)
        updates = (payload.dict(exclude_unset=True) if dump_method is None
                   else dump_method(exclude_unset=True))
        if "name" in updates and updates["name"] is None:
            raise HTTPException(status_code=422, detail="name cannot be null")
        if ("lat" in updates) != ("lon" in updates):
            raise HTTPException(status_code=422, detail="lat and lon must be updated together")

        name = updates.get("name", existing["name"])
        parent_id = updates.get("parent_id", existing["parent_id"])
        lat = updates.get("lat", existing["lat"])
        lon = updates.get("lon", existing["lon"])
        if (lat is None) != (lon is None):
            raise HTTPException(status_code=422, detail="lat and lon must be provided together")
        validate_place_parent(conn, parent_id, place_id)
        ensure_unique_place_name(conn, name, parent_id, place_id)
        conn.execute(
            """
            UPDATE places
            SET name = ?, parent_id = ?, lat = ?, lon = ?, updated_at = ?
            WHERE id = ?
            """,
            (name, parent_id, lat, lon, datetime.now().astimezone().isoformat(), place_id),
        )
        conn.commit()
        return place_to_response(conn, fetch_place_by_id(place_id, conn))
    finally:
        conn.close()


@router.get("/places/{place_id}")
async def get_place(place_id: int):
    """Return one place, its children, and memories from its whole subtree."""
    conn = get_db_connection()
    try:
        row = fetch_place_by_id(place_id, conn)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Place {place_id} not found")
        place = place_to_response(conn, row)
        child_rows = conn.execute(
            "SELECT id, name, parent_id, lat, lon, created_at, updated_at "
            "FROM places WHERE parent_id = ? ORDER BY LOWER(name)",
            (place_id,),
        ).fetchall()
        descendant_ids = place_descendant_ids(conn, place_id)
        placeholders = ",".join("?" for _ in descendant_ids)
        memory_rows = conn.execute(
            f"""
            SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, image_path,
                   text_note, note_created_at, note_updated_at, user_note, bepo_summary,
                   tags, mood, place_hint, context_type, shopping_status,
                   shopping_status_updated_at, place_id
            FROM memories
            WHERE place_id IN ({placeholders})
            ORDER BY COALESCE(taken_at, ts) DESC
            """,
            descendant_ids,
        ).fetchall()
        return {
            **place,
            "children": [place_to_response(conn, child) for child in child_rows],
            "memories": [row_to_memory_response(memory) for memory in memory_rows],
        }
    finally:
        conn.close()

@router.post("/search")
async def search_memories(
    query: str = Form(...),
    top_k: int = Form(5),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    place_id: Optional[int] = Form(None),
):
    """
    Search memories by text query.
    Returns up to *top_k* matches sorted by score descending.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty or whitespace")
    if not (1 <= top_k <= 20):
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")
    if lat is not None and not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="lat must be between -90 and 90")
    if lon is not None and not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="lon must be between -180 and 180")

    try:
        matches, filters, needs_location = search_memory_matches_with_filters(
            query.strip(), top_k, lat, lon, place_id
        )

        if filters["place_options"]:
            return {
                "status": "needs_place",
                "message": "Choose which saved place you mean",
                "filters": public_memory_filters(filters),
                "place_options": filters["place_options"],
                "matches": [],
            }

        if filters["suggested_place_name"]:
            return {
                "status": "no_results",
                "message": f"No saved place named {filters['suggested_place_name']}",
                "filters": public_memory_filters(filters),
                "suggested_place_name": filters["suggested_place_name"],
                "matches": [],
            }

        if needs_location:
            return {
                "status": "needs_location",
                "message": "Current location is needed for nearby results",
                "filters": public_memory_filters(filters),
                "matches": [],
            }

        if not matches:
            return {
                "status": "no_results",
                "message": "No memories found in database",
                "filters": public_memory_filters(filters),
                "matches": [],
            }

        return {
            "status": "success",
            "query": query,
            "count": len(matches),
            "filters": public_memory_filters(filters),
            "matches": matches,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask a natural question about saved memories.
    Returns a simple deterministic answer built from the top matching memory.
    """
    try:
        matches, filters, needs_location = search_memory_matches_with_filters(
            request.message.strip(),
            request.top_k,
            request.lat,
            request.lon,
            request.place_id,
            request.detect_places,
        )
        applied_filters = public_memory_filters(filters)

        if filters["place_options"]:
            place_name = filters["place_options"][0]["name"]
            return {
                "status": "needs_place",
                "message": request.message,
                "answer": f"I found more than one {place_name}. Which place did you mean?",
                "filters": applied_filters,
                "place_options": filters["place_options"],
                "memories": [],
            }

        if filters["suggested_place_name"]:
            suggested_name = filters["suggested_place_name"]
            return {
                "status": "no_results",
                "message": request.message,
                "answer": f"I do not have a place named {suggested_name} yet. Is this a place you want to create?",
                "filters": applied_filters,
                "suggested_place_name": suggested_name,
                "memories": [],
            }

        if needs_location:
            return {
                "status": "needs_location",
                "message": request.message,
                "answer": "I need your current location to sort these memories nearby. You can allow location access and try again.",
                "filters": applied_filters,
                "memories": [],
            }

        if not matches:
            place = filters.get("place")
            return {
                "status": "no_results",
                "message": request.message,
                "answer": (
                    f"I could not find any memories in {place['path_label']} yet."
                    if place
                    else "I do not have any memories saved yet."
                    if not filters["_has_memories"]
                    else "I could not find any memories matching those details."
                    if applied_filters
                    else "I do not have any memories saved yet."
                ),
                "filters": applied_filters,
                "memories": [],
            }

        answer = build_filtered_chat_answer(filters, len(matches), matches[0]) if applied_filters else build_chat_answer(matches[0])

        return {
            "status": "success",
            "message": request.message,
            "answer": answer,
            "count": len(matches),
            "filters": applied_filters,
            "memories": matches,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")


@router.get("/memories")
async def list_memories():
    """Return all saved memories, newest first. Embeddings are not included."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, image_path, "
            "text_note, note_created_at, note_updated_at, user_note, bepo_summary, tags, mood, place_hint, "
            "context_type, shopping_status, shopping_status_updated_at, place_id "
            "FROM memories ORDER BY COALESCE(taken_at, ts) DESC"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [row_to_memory_response(row) for row in rows]


@router.get("/memory/{memory_id}")
async def get_memory(memory_id: int):
    """Return a single memory by id. Returns 404 if not found."""
    row = fetch_memory_by_id(memory_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return row_to_memory_response(row)


@router.get("/memory/{memory_id}/status-history")
async def get_memory_status_history(memory_id: int):
    """Return shopping-status changes for a memory, oldest first."""
    if fetch_memory_by_id(memory_id) is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, changed_at FROM memory_status_history "
            "WHERE memory_id = ? ORDER BY changed_at ASC, id ASC",
            (memory_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


@router.get("/image/{memory_id}")
async def get_image(memory_id: int):
    """Serve the image file associated with a memory. Returns 404 if not found."""
    row = fetch_memory_by_id(memory_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    image_path = row["image_path"]
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail=f"Image file not found for memory {memory_id}")
    return FileResponse(image_path)

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": "Bepo",
        "version": "1.0",
        "description": "Memory storage and search with image and text embeddings",
        "endpoints": {
            "POST /memory": "Store a new memory with photo, note, and GPS",
            "PATCH /memory/{id}/metadata": "Update editable memory metadata and refresh text embeddings",
            "GET /memories": "List all memories, newest first",
            "GET /memory/{id}": "Get a single memory by id",
            "GET /memory/{id}/status-history": "Show shopping-status changes for a memory",
            "GET /places": "List the manual place forest",
            "POST /places": "Create a root place or nested subplace",
            "PATCH /places/{id}": "Rename, move, or pin a place",
            "GET /places/{id}": "Open one place branch and its memories",
            "GET /places/suggestions": "Suggest existing nearby pins without auto-assigning",
            "GET /image/{id}": "Serve the image for a memory",
            "POST /search": "Search with exact tags, moods, context, shopping stage, and nearby distance",
            "POST /chat": "Ask naturally with structured filters and optional current location",
        },
    }


app.include_router(router)


@app.get("/health", include_in_schema=False)
async def health():
    """Return a public liveness response for deployment health checks."""
    return {
        "status": "ok",
        "visual_search": USE_CLIP and model is not None,
        "embedding_size": active_embedding_size(),
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )

