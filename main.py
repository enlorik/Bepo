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


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag with conservative parsing."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# CLIP is opt-in so small hosted deployments can start quickly and reliably.
# Install requirements-clip.txt and set BEPO_ENABLE_CLIP=1 to enable it.
USE_CLIP = env_flag("BEPO_ENABLE_CLIP")
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


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=3, ge=1, le=10)
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lon: Optional[float] = Field(default=None, ge=-180, le=180)

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
    cursor = conn.cursor()
    
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
            shopping_status_updated_at DATETIME
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
            "shopping_status_updated_at "
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


def parse_memory_query(query: str, known_moods: set[str]) -> dict:
    """Extract exact filters from a conversational memory query."""
    working = query.casefold()
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
    return {
        "tags": list(dict.fromkeys(tags)),
        "moods": list(dict.fromkeys(moods)),
        "context_type": context_type,
        "shopping_status": shopping_status,
        "nearby": nearby,
        "semantic_query": semantic_query,
    }


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
) -> tuple[list, dict, bool]:
    """Return matches, parsed filters, and whether a nearby query needs GPS."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, ts, taken_at, taken_at_source, lat, lon, location_source, "
            "image_path, image_emb, text_note, text_emb, note_created_at, note_updated_at, "
            "user_note, bepo_summary, tags, mood, place_hint, context_type, shopping_status, "
            "shopping_status_updated_at "
            "FROM memories"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return [], parse_memory_query(query, set()), False

    known_moods = {
        mood
        for row in rows
        for mood in split_metadata_values(row["mood"])
    }
    known_moods.update({"calm", "cozy", "happy", "excited", "nostalgic", "safe", "romantic", "sad", "anxious"})
    filters = parse_memory_query(query, known_moods)
    needs_location = filters["nearby"] and (current_lat is None or current_lon is None)
    if needs_location:
        return [], filters, True

    semantic_query = filters["semantic_query"]
    query_emb = get_text_embedding(semantic_query) if semantic_query else None

    scored = []
    for row in rows:
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

        image_score = 1.0
        if query_emb is not None:
            image_emb = deserialize_embedding(row["image_emb"])
            image_score = cosine_similarity(query_emb, image_emb)

        text_score = -1.0
        if query_emb is not None and row["text_emb"] is not None:
            text_emb_arr = deserialize_embedding(row["text_emb"])
            text_score = cosine_similarity(query_emb, text_emb_arr)

        score = max(image_score, text_score)
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
            "lat": lat,
            "lon": lon,
            "location_source": row["location_source"],
            "map_url": build_map_url(lat, lon),
            "distance_km": distance_km,
            "score": score,
        })

    if filters["nearby"]:
        scored.sort(key=lambda item: (item["distance_km"], -item["score"]))
    elif semantic_query:
        scored.sort(key=lambda item: item["score"], reverse=True)
    else:
        scored.sort(key=lambda item: item["taken_at"] or item["timestamp"], reverse=True)
    return scored[:top_k], filters, False


def search_memory_matches(query: str, top_k: int) -> list:
    """Backward-compatible semantic/structured search helper."""
    matches, _, _ = search_memory_matches_with_filters(query, top_k)
    return matches


def public_memory_filters(filters: dict) -> dict:
    """Return only user-facing filters, omitting internal semantic-query text."""
    return {key: value for key, value in filters.items() if key != "semantic_query" and value not in (None, [], False)}


def build_filtered_chat_answer(filters: dict, count: int, top: Optional[dict] = None) -> str:
    """Describe an exact filtered result set in friendly language."""
    nearby = filters.get("nearby")
    noun = "memory" if count == 1 else "memories"
    ordering = " nearby, closest first" if nearby else ""
    answer = f"I found {count} matching {noun}{ordering}."
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
                    shopping_status, shopping_status_updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    merged_values = {
        "text_note": existing_row["text_note"],
        "user_note": existing_row["user_note"],
        "bepo_summary": existing_row["bepo_summary"],
        "tags": existing_row["tags"],
        "mood": existing_row["mood"],
        "place_hint": existing_row["place_hint"],
        "context_type": existing_row["context_type"],
        "shopping_status": existing_row["shopping_status"],
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
                text_emb = ?, note_created_at = ?, note_updated_at = ?
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
            "context_type, shopping_status, shopping_status_updated_at "
            "FROM memories WHERE id = ?",
            (memory_id,),
        )
        updated_row = cursor.fetchone()
    finally:
        conn.close()

    return row_to_memory_response(updated_row)

@router.post("/search")
async def search_memories(
    query: str = Form(...),
    top_k: int = Form(5),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
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
            query.strip(), top_k, lat, lon
        )

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
            request.message.strip(), request.top_k, request.lat, request.lon
        )
        applied_filters = public_memory_filters(filters)

        if needs_location:
            return {
                "status": "needs_location",
                "message": request.message,
                "answer": "I need your current location to sort these memories nearby. You can allow location access and try again.",
                "filters": applied_filters,
                "memories": [],
            }

        if not matches:
            return {
                "status": "no_results",
                "message": request.message,
                "answer": "I could not find any memories matching those details." if applied_filters
                else "I do not have any memories saved yet.",
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
            "context_type, shopping_status, shopping_status_updated_at "
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
        "version": "0.8",
        "description": "Memory storage and search with image and text embeddings",
        "endpoints": {
            "POST /memory": "Store a new memory with photo, note, and GPS",
            "PATCH /memory/{id}/metadata": "Update editable memory metadata and refresh text embeddings",
            "GET /memories": "List all memories, newest first",
            "GET /memory/{id}": "Get a single memory by id",
            "GET /memory/{id}/status-history": "Show shopping-status changes for a memory",
            "GET /image/{id}": "Serve the image for a memory",
            "POST /search": "Search with exact tags, moods, context, shopping stage, and nearby distance",
            "POST /chat": "Ask naturally with structured filters and optional current location",
        },
    }


app.include_router(router)


@app.get("/health", include_in_schema=False)
async def health():
    """Return a public liveness response for deployment health checks."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )

