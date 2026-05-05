"""
Basic tests for the Bepo API.

CLIP model loading is suppressed in tests by patching USE_CLIP to False and
replacing init_model with a no-op so no network calls are made.
All tests use a temporary SQLite database via a fixture that patches DB_PATH.
"""
import io
import pytest
import main as app_module
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Point DB_PATH / IMAGES_DIR at temp paths and ensure CLIP is disabled
    so tests never attempt a model download.
    """
    # Suppress CLIP – must be patched before init_db/lifespan runs
    monkeypatch.setattr(app_module, "USE_CLIP", False)
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "processor", None)
    # Replace init_model with a no-op so lifespan never touches the network
    monkeypatch.setattr(app_module, "init_model", lambda: None)

    db_file = str(tmp_path / "test_memories.db")
    monkeypatch.setattr(app_module, "DB_PATH", db_file)

    img_dir = str(tmp_path / "images")
    monkeypatch.setattr(app_module, "IMAGES_DIR", img_dir)

    app_module.init_db()
    yield


@pytest.fixture()
def client(isolated_db):
    with TestClient(app_module.app) as c:
        yield c


# ── helpers ────────────────────────────────────────────────────────────────

def _tiny_jpeg() -> bytes:
    """Return a minimal valid JPEG image as bytes."""
    img = app_module.Image.new("RGB", (8, 8), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── tests ──────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_app_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["app"] == "Bepo"
        assert "endpoints" in data


class TestListMemories:
    def test_empty_database_returns_empty_list(self, client):
        r = client.get("/memories")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_memories_after_insert(self, client):
        # Insert one memory
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "hello"},
        )
        r = client.get("/memories")
        assert r.status_code == 200
        memories = r.json()
        assert len(memories) == 1
        m = memories[0]
        assert "id" in m
        assert "timestamp" in m
        assert "image_url" in m
        assert "image_emb" not in m
        assert "text_emb" not in m


class TestGetMemory:
    def test_404_for_missing_id(self, client):
        r = client.get("/memory/9999")
        assert r.status_code == 404

    def test_returns_memory_by_id(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "test note", "lat": "1.23", "lon": "4.56"},
        )
        memory_id = resp.json()["memory_id"]
        r = client.get(f"/memory/{memory_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == memory_id
        assert data["note"] == "test note"
        assert data["lat"] == pytest.approx(1.23)
        assert data["lon"] == pytest.approx(4.56)
        assert "image_emb" not in data
        assert "text_emb" not in data


class TestGetImage:
    def test_404_for_missing_memory(self, client):
        r = client.get("/image/9999")
        assert r.status_code == 404

    def test_returns_image_bytes(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
        )
        memory_id = resp.json()["memory_id"]
        r = client.get(f"/image/{memory_id}")
        assert r.status_code == 200


class TestSearch:
    def test_no_results_on_empty_database(self, client):
        r = client.post("/search", data={"query": "sunset"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "no_results"
        assert data["matches"] == []

    def test_returns_matches(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "sunset at the beach"},
        )
        r = client.post("/search", data={"query": "sunset", "top_k": "3"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["count"] >= 1
        match = data["matches"][0]
        assert "id" in match
        assert "score" in match
        assert "image_url" in match

    def test_empty_query_rejected(self, client):
        r = client.post("/search", data={"query": "   "})
        assert r.status_code == 422

    def test_top_k_out_of_range_rejected(self, client):
        r = client.post("/search", data={"query": "test", "top_k": "0"})
        assert r.status_code == 422
        r2 = client.post("/search", data={"query": "test", "top_k": "21"})
        assert r2.status_code == 422


class TestBuildMapUrl:
    def test_returns_url_when_coords_present(self):
        url = app_module.build_map_url(34.0522, -118.2437)
        assert url is not None
        assert "34.0522" in url
        assert "-118.2437" in url
        assert url.startswith("https://www.google.com/maps")

    def test_returns_none_when_lat_missing(self):
        assert app_module.build_map_url(None, -118.2437) is None

    def test_returns_none_when_lon_missing(self):
        assert app_module.build_map_url(34.0522, None) is None

    def test_returns_none_when_both_missing(self):
        assert app_module.build_map_url(None, None) is None
