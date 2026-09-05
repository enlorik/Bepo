"""
Basic tests for the Bepo API.

CLIP model loading is suppressed in tests by patching USE_CLIP to False and
replacing init_model with a no-op so no network calls are made.
All tests use a temporary SQLite database via a fixture that patches DB_PATH.
"""
import io
import sqlite3
import numpy as np
import pytest
import main as app_module
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """
    Point DB_PATH / IMAGES_DIR at temp paths and ensure CLIP is disabled
    so tests never attempt a model download.
    """
    # Force offline mode and suppress CLIP before init_db/lifespan runs
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(app_module, "USE_CLIP", False)
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "processor", None)
    monkeypatch.setattr(app_module, "API_KEY", None)
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


def _fetch_memory_row(memory_id: int):
    conn = sqlite3.connect(app_module.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text_note, user_note, bepo_summary, tags, mood, place_hint, context_type, "
            "shopping_status, shopping_status_updated_at, text_emb, note_created_at, note_updated_at "
            "FROM memories WHERE id = ?",
            (memory_id,),
        )
        return cursor.fetchone()
    finally:
        conn.close()


# ── tests ──────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_app_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["app"] == "Bepo"
        assert "endpoints" in data

    def test_health_is_public(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "API_KEY", "test-secret")
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_configured_api_key_protects_application_routes(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "API_KEY", "test-secret")

        missing = client.get("/")
        invalid = client.get("/", headers={"X-API-Key": "wrong"})
        valid = client.get("/", headers={"X-API-Key": "test-secret"})

        assert missing.status_code == 401
        assert invalid.status_code == 401
        assert valid.status_code == 200


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
        assert "user_note" in m
        assert "bepo_summary" in m
        assert "tags" in m
        assert "mood" in m
        assert "place_hint" in m
        assert "added_at" in m
        assert "taken_at" in m
        assert "note_created_at" in m
        assert "location_source" in m
        assert m["context_type"] == "unknown"
        assert m["shopping_status"] is None
        assert m["shopping_status_updated_at"] is None

    def test_old_minimal_post_memory_still_works(self, client):
        r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "minimal"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        memory_id = data["memory_id"]

        get_r = client.get(f"/memory/{memory_id}")
        assert get_r.status_code == 200
        memory = get_r.json()
        assert memory["note"] == "minimal"
        assert memory["user_note"] is None
        assert memory["mood"] is None
        assert memory["tags"] is None
        assert memory["place_hint"] is None
        assert memory["context_type"] == "unknown"
        assert memory["shopping_status"] is None

    def test_gallery_is_ordered_by_event_date_when_known(self, client):
        newer_event = client.post(
            "/memory",
            files={"photo": ("newer.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"taken_at": "2024-08-01T12:00:00", "taken_at_source": "photo"},
        ).json()["memory_id"]
        older_event_added_later = client.post(
            "/memory",
            files={"photo": ("older.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"taken_at": "2018-03-02T09:00:00", "taken_at_source": "photo"},
        ).json()["memory_id"]

        memories = client.get("/memories").json()
        assert [memory["id"] for memory in memories] == [newer_event, older_event_added_later]


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
        assert data["user_note"] is None
        assert data["bepo_summary"] is None
        assert data["tags"] is None
        assert data["mood"] is None
        assert data["place_hint"] is None
        assert data["context_type"] == "physical"
        assert data["shopping_status"] is None

    def test_returns_new_metadata_fields(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "cafe",
                "user_note": "cat on chair",
                "bepo_summary": "quiet cat cafe",
                "tags": "cafe,cat,cozy",
                "mood": "calm",
                "place_hint": "near red couch",
            },
        )
        memory_id = resp.json()["memory_id"]
        r = client.get(f"/memory/{memory_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["note"] == "cafe"
        assert data["user_note"] == "cat on chair"
        assert data["bepo_summary"] == "quiet cat cafe"
        assert data["tags"] == "cafe,cat,cozy"
        assert data["mood"] == "calm"
        assert data["place_hint"] == "near red couch"

    def test_accepts_online_context_and_initial_shopping_status(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "striped cardigan", "context_type": "online", "shopping_status": "want"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["context_type"] == "online"
        assert data["shopping_status"] == "want"
        assert data["shopping_status_updated_at"] is not None

        history = client.get(f"/memory/{data['memory_id']}/status-history")
        assert history.status_code == 200
        assert history.json() == [{"status": "want", "changed_at": data["shopping_status_updated_at"]}]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("context_type", "somewhere"), ("shopping_status", "maybe")],
    )
    def test_rejects_invalid_structured_metadata(self, client, field, value):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={field: value},
        )
        assert resp.status_code == 422

    def test_keeps_photo_time_separate_from_added_and_note_times(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "an older summer day",
                "taken_at": "2021-07-18T14:20:00",
                "taken_at_source": "photo",
                "lat": "54.3520",
                "lon": "18.6466",
                "location_source": "photo",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["taken_at"] == "2021-07-18T14:20:00"
        assert data["taken_at_source"] == "photo"
        assert data["location_source"] == "photo"
        assert data["added_at"] == data["timestamp"]
        assert data["note_created_at"] == data["added_at"]
        assert data["note_updated_at"] == data["added_at"]

    def test_rejects_invalid_photo_time(self, client):
        resp = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"taken_at": "not-a-date"},
        )
        assert resp.status_code == 422
        assert "taken_at" in resp.json()["detail"]


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


class TestPatchMemoryMetadata:
    def test_updates_selected_metadata_fields(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "original note", "bepo_summary": "keep summary"},
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={
                "user_note": "new user note",
                "mood": "excited",
                "tags": "park,dog",
                "place_hint": "near the fountain",
            },
        )

        assert patch_r.status_code == 200
        data = patch_r.json()
        assert data["id"] == memory_id
        assert data["note"] == "original note"
        assert data["user_note"] == "new user note"
        assert data["bepo_summary"] == "keep summary"
        assert data["tags"] == "park,dog"
        assert data["mood"] == "excited"
        assert data["place_hint"] == "near the fountain"
        assert "text_emb" not in data
        assert "image_emb" not in data

    def test_updates_context_and_tracks_shopping_status_history(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "a lamp", "shopping_status": "want"},
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"context_type": "online", "shopping_status": "bought"},
        )

        assert patch_r.status_code == 200
        data = patch_r.json()
        assert data["context_type"] == "online"
        assert data["shopping_status"] == "bought"
        assert data["shopping_status_updated_at"] is not None

        history = client.get(f"/memory/{memory_id}/status-history").json()
        assert [entry["status"] for entry in history] == ["want", "bought"]

    def test_clearing_shopping_status_is_recorded(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"shopping_status": "ordered"},
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(f"/memory/{memory_id}/metadata", json={"shopping_status": None})

        assert patch_r.status_code == 200
        assert patch_r.json()["shopping_status"] is None
        history = client.get(f"/memory/{memory_id}/status-history").json()
        assert [entry["status"] for entry in history] == ["ordered", None]

    @pytest.mark.parametrize(
        ("payload", "field"),
        [({"context_type": "nearby"}, "context_type"), ({"shopping_status": "soon"}, "shopping_status")],
    )
    def test_rejects_invalid_edit_values(self, client, payload, field):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
        )
        memory_id = create_r.json()["memory_id"]
        patch_r = client.patch(f"/memory/{memory_id}/metadata", json=payload)
        assert patch_r.status_code == 422
        assert field in str(patch_r.json())

    def test_records_when_a_note_is_added_later(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
        )
        memory_id = create_r.json()["memory_id"]
        assert create_r.json()["note_created_at"] is None

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"note": "remembered this later"},
        )

        assert patch_r.status_code == 200
        data = patch_r.json()
        assert data["note_created_at"] is not None
        assert data["note_updated_at"] == data["note_created_at"]

    def test_omitted_fields_preserve_existing_values(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "original note",
                "user_note": "original user note",
                "bepo_summary": "original summary",
                "tags": "old,tags",
                "mood": "calm",
                "place_hint": "old place",
            },
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"mood": "joyful"},
        )

        assert patch_r.status_code == 200
        data = patch_r.json()
        assert data["note"] == "original note"
        assert data["user_note"] == "original user note"
        assert data["bepo_summary"] == "original summary"
        assert data["tags"] == "old,tags"
        assert data["mood"] == "joyful"
        assert data["place_hint"] == "old place"

    def test_null_clears_a_field(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "note", "mood": "calm"},
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"mood": None},
        )

        assert patch_r.status_code == 200
        assert patch_r.json()["mood"] is None

    def test_text_embedding_is_recomputed_after_update(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "first note"},
        )
        memory_id = create_r.json()["memory_id"]
        before_row = _fetch_memory_row(memory_id)
        before_emb = app_module.deserialize_embedding(before_row["text_emb"])

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"user_note": "updated metadata"},
        )

        assert patch_r.status_code == 200
        after_row = _fetch_memory_row(memory_id)
        after_emb = app_module.deserialize_embedding(after_row["text_emb"])
        expected_emb = app_module.get_text_embedding("first note\nupdated metadata")

        assert not np.allclose(before_emb, after_emb)
        assert np.allclose(after_emb, expected_emb)

    def test_clearing_all_text_fields_sets_text_embedding_to_null(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "note",
                "user_note": "user",
                "bepo_summary": "summary",
                "tags": "tag1,tag2",
                "mood": "happy",
                "place_hint": "somewhere",
            },
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={
                "note": None,
                "user_note": None,
                "bepo_summary": None,
                "tags": None,
                "mood": None,
                "place_hint": None,
            },
        )

        assert patch_r.status_code == 200
        row = _fetch_memory_row(memory_id)
        assert row["text_note"] is None
        assert row["user_note"] is None
        assert row["bepo_summary"] is None
        assert row["tags"] is None
        assert row["mood"] is None
        assert row["place_hint"] is None
        assert row["text_emb"] is None

    def test_missing_memory_returns_404(self, client):
        patch_r = client.patch("/memory/9999/metadata", json={"mood": "calm"})
        assert patch_r.status_code == 404

    def test_updated_metadata_appears_in_get_memory(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "old note"},
        )
        memory_id = create_r.json()["memory_id"]

        client.patch(
            f"/memory/{memory_id}/metadata",
            json={"user_note": "fresh detail", "place_hint": "library stairs"},
        )

        get_r = client.get(f"/memory/{memory_id}")
        assert get_r.status_code == 200
        data = get_r.json()
        assert data["user_note"] == "fresh detail"
        assert data["place_hint"] == "library stairs"

    def test_search_finds_memory_using_new_metadata_after_patch(self, client):
        create_r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "plain photo"},
        )
        memory_id = create_r.json()["memory_id"]

        patch_r = client.patch(
            f"/memory/{memory_id}/metadata",
            json={"place_hint": "purple bicycle rack"},
        )
        assert patch_r.status_code == 200

        search_r = client.post("/search", data={"query": "purple bicycle rack", "top_k": "3"})
        assert search_r.status_code == 200
        data = search_r.json()
        assert data["status"] == "success"
        assert data["count"] >= 1
        assert data["matches"][0]["id"] == memory_id
        assert data["matches"][0]["place_hint"] == "purple bicycle rack"


class TestPlaces:
    def test_creates_a_manual_place_forest_with_inherited_pin(self, client):
        home = client.post(
            "/places", json={"name": "Home", "lat": 52.23, "lon": 21.01}
        )
        assert home.status_code == 200
        home_data = home.json()
        kitchen = client.post(
            "/places", json={"name": "Kitchen", "parent_id": home_data["id"]}
        )
        assert kitchen.status_code == 200
        kitchen_data = kitchen.json()

        assert kitchen_data["path_label"] == "Home › Kitchen"
        assert kitchen_data["effective_lat"] == pytest.approx(52.23)
        assert kitchen_data["effective_lon"] == pytest.approx(21.01)
        assert kitchen_data["pin_inherited"] is True

        places = client.get("/places").json()
        assert [place["path_label"] for place in places] == ["Home", "Home › Kitchen"]
        assert places[0]["child_count"] == 1

    def test_rejects_missing_parent_and_duplicate_sibling_name(self, client):
        missing = client.post("/places", json={"name": "Bedroom", "parent_id": 999})
        assert missing.status_code == 422

        home_id = client.post("/places", json={"name": "Home"}).json()["id"]
        first = client.post("/places", json={"name": "Bedroom", "parent_id": home_id})
        duplicate = client.post("/places", json={"name": "bedroom", "parent_id": home_id})
        assert first.status_code == 200
        assert duplicate.status_code == 409

    def test_prevents_reparenting_cycle(self, client):
        home_id = client.post("/places", json={"name": "Home"}).json()["id"]
        room_id = client.post(
            "/places", json={"name": "Room", "parent_id": home_id}
        ).json()["id"]

        response = client.patch(f"/places/{home_id}", json={"parent_id": room_id})

        assert response.status_code == 422
        assert "cannot be inside" in response.json()["detail"]

    def test_assigns_memory_and_parent_includes_nested_memories(self, client):
        home_id = client.post("/places", json={"name": "Home"}).json()["id"]
        kitchen_id = client.post(
            "/places", json={"name": "Kitchen", "parent_id": home_id}
        ).json()["id"]
        memory_id = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "tea shelf"},
        ).json()["memory_id"]

        assigned = client.patch(
            f"/memory/{memory_id}/metadata", json={"place_id": kitchen_id}
        )

        assert assigned.status_code == 200
        assert assigned.json()["place_id"] == kitchen_id
        kitchen = client.get(f"/places/{kitchen_id}").json()
        home = client.get(f"/places/{home_id}").json()
        assert [memory["id"] for memory in kitchen["memories"]] == [memory_id]
        assert [memory["id"] for memory in home["memories"]] == [memory_id]
        assert home["memory_count"] == 1
        assert home["direct_memory_count"] == 0

    def test_nearby_suggestions_only_return_existing_explicit_pins(self, client):
        cafe_id = client.post(
            "/places", json={"name": "Cat Cafe", "lat": 50.061, "lon": 19.938}
        ).json()["id"]
        child_id = client.post(
            "/places", json={"name": "Upstairs", "parent_id": cafe_id}
        ).json()["id"]
        client.post(
            "/places", json={"name": "Far Cafe", "lat": 51.1, "lon": 19.9}
        )
        memory_id = client.post(
            "/memory",
            files={"photo": ("nearby.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"lat": "50.06", "lon": "19.94"},
        ).json()["memory_id"]

        suggestions = client.get(
            "/places/suggestions",
            params={"lat": 50.06, "lon": 19.94, "radius_m": 1000},
        )

        assert suggestions.status_code == 200
        data = suggestions.json()
        assert [place["id"] for place in data] == [cafe_id]
        assert child_id not in [place["id"] for place in data]
        assert data[0]["distance_m"] < 1000
        assert client.get(f"/memory/{memory_id}").json()["place_id"] is None

    def test_memory_assignment_rejects_unknown_place_and_can_be_cleared(self, client):
        place_id = client.post("/places", json={"name": "Library"}).json()["id"]
        memory_id = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"place_id": str(place_id)},
        ).json()["memory_id"]
        assert client.get(f"/memory/{memory_id}").json()["place_id"] == place_id

        invalid = client.patch(f"/memory/{memory_id}/metadata", json={"place_id": 999})
        assert invalid.status_code == 422
        cleared = client.patch(f"/memory/{memory_id}/metadata", json={"place_id": None})
        assert cleared.status_code == 200
        assert cleared.json()["place_id"] is None


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
        assert "user_note" in match
        assert "bepo_summary" in match
        assert "tags" in match
        assert "mood" in match
        assert "place_hint" in match

    def test_post_with_mood_tags_place_hint_works(self, client):
        r = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "mood": "calm",
                "tags": "cat,cafe",
                "place_hint": "near the red couch hallway",
            },
        )
        assert r.status_code == 200
        memory_id = r.json()["memory_id"]

        get_r = client.get(f"/memory/{memory_id}")
        assert get_r.status_code == 200
        memory = get_r.json()
        assert memory["mood"] == "calm"
        assert memory["tags"] == "cat,cafe"
        assert memory["place_hint"] == "near the red couch hallway"

    def test_search_finds_memory_from_richer_metadata(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "user_note": "cat sleeping by the window",
                "bepo_summary": "cozy cafe",
                "tags": "cute barista place",
                "mood": "calm",
                "place_hint": "that weird hallway near the red couch",
            },
        )
        r = client.post("/search", data={"query": "red couch hallway", "top_k": "3"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["count"] >= 1
        top = data["matches"][0]
        assert top["place_hint"] == "that weird hallway near the red couch"

    def test_empty_query_rejected(self, client):
        r = client.post("/search", data={"query": "   "})
        assert r.status_code == 422

    def test_top_k_out_of_range_rejected(self, client):
        r = client.post("/search", data={"query": "test", "top_k": "0"})
        assert r.status_code == 422
        r2 = client.post("/search", data={"query": "test", "top_k": "21"})
        assert r2.status_code == 422

    def test_hashtag_filter_is_exact(self, client):
        cafe_id = client.post(
            "/memory",
            files={"photo": ("cafe.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "window table", "tags": "cafe"},
        ).json()["memory_id"]
        client.post(
            "/memory",
            files={"photo": ("other.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "office machine", "tags": "cafeteria"},
        )

        data = client.post("/search", data={"query": "#cafe"}).json()

        assert data["status"] == "success"
        assert data["filters"] == {"tags": ["cafe"]}
        assert [memory["id"] for memory in data["matches"]] == [cafe_id]

    def test_structured_filters_are_combined(self, client):
        matching_id = client.post(
            "/memory",
            files={"photo": ("match.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "green cardigan",
                "tags": "shopping",
                "mood": "calm,hopeful",
                "context_type": "online",
                "shopping_status": "bought",
            },
        ).json()["memory_id"]
        client.post(
            "/memory",
            files={"photo": ("wrong.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "note": "blue cardigan",
                "tags": "shopping",
                "mood": "excited",
                "context_type": "online",
                "shopping_status": "want",
            },
        )

        data = client.post("/search", data={"query": "#shopping calm bought online"}).json()

        assert data["status"] == "success"
        assert [memory["id"] for memory in data["matches"]] == [matching_id]
        assert data["filters"] == {
            "tags": ["shopping"],
            "moods": ["calm"],
            "context_type": "online",
            "shopping_status": "bought",
        }

    def test_nearby_requires_current_location(self, client):
        client.post(
            "/memory",
            files={"photo": ("cafe.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "cafe", "lat": "52.23", "lon": "21.01"},
        )

        data = client.post("/search", data={"query": "#cafe nearby"}).json()

        assert data["status"] == "needs_location"
        assert data["filters"] == {"tags": ["cafe"], "nearby": True}

    def test_nearby_excludes_online_and_sorts_closest_first(self, client):
        far_id = client.post(
            "/memory",
            files={"photo": ("far.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "cafe", "lat": "52.30", "lon": "21.01"},
        ).json()["memory_id"]
        near_id = client.post(
            "/memory",
            files={"photo": ("near.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "cafe", "lat": "52.231", "lon": "21.011"},
        ).json()["memory_id"]
        client.post(
            "/memory",
            files={"photo": ("online.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "cafe", "context_type": "online", "lat": "52.2301", "lon": "21.0101"},
        )

        data = client.post(
            "/search",
            data={"query": "#cafe nearby", "lat": "52.23", "lon": "21.01"},
        ).json()

        assert [memory["id"] for memory in data["matches"]] == [near_id, far_id]
        assert data["matches"][0]["distance_km"] < data["matches"][1]["distance_km"]
        assert all(memory["context_type"] != "online" for memory in data["matches"])


class TestChat:
    def test_no_results_on_empty_database(self, client):
        r = client.post("/chat", json={"message": "Where was the calm cafe?"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "no_results"
        assert data["memories"] == []
        assert data["answer"] == "I do not have any memories saved yet."

    def test_rejects_empty_message(self, client):
        r = client.post("/chat", json={"message": ""})
        assert r.status_code == 422

    def test_rejects_whitespace_only_message(self, client):
        r = client.post("/chat", json={"message": "   "})
        assert r.status_code == 422

    def test_rejects_top_k_below_min(self, client):
        r = client.post("/chat", json={"message": "test", "top_k": 0})
        assert r.status_code == 422

    def test_rejects_top_k_above_max(self, client):
        r = client.post("/chat", json={"message": "test", "top_k": 11})
        assert r.status_code == 422

    def test_returns_relevant_memory_after_creating_one(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "quiet cafe with a cat", "mood": "calm", "tags": "cafe,cat"},
        )
        r = client.post("/chat", json={"message": "cafe with a cat"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["count"] >= 1
        assert len(data["memories"]) >= 1
        assert data["memories"][0]["note"] == "quiet cafe with a cat"

    def test_answer_includes_useful_metadata_from_top_memory(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={
                "mood": "calm",
                "tags": "cafe,cat,cozy",
                "place_hint": "near the red couch hallway",
                "taken_at": "2021-07-18T14:20:00",
                "taken_at_source": "photo",
            },
        )
        r = client.post("/chat", json={"message": "calm cafe with a cat"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        answer = data["answer"]
        # Answer should reference the place hint and/or mood/tags
        assert any(kw in answer for kw in ["red couch", "calm", "cafe", "cat", "cozy"])
        assert "July 18, 2021" in answer

    def test_answer_formats_multiple_moods_as_separate_details(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "window seat", "mood": "calm,cozy", "tags": "cafe"},
        )
        r = client.post("/chat", json={"message": "window seat"})
        assert r.status_code == 200
        answer = r.json()["answer"]
        assert "calm, cozy" in answer
        assert "calm,cozy" not in answer

    def test_does_not_expose_embeddings(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "test memory"},
        )
        r = client.post("/chat", json={"message": "test"})
        assert r.status_code == 200
        data = r.json()
        for memory in data["memories"]:
            assert "image_emb" not in memory
            assert "text_emb" not in memory

    def test_search_still_works_after_helper_refactor(self, client):
        client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "ocean at dusk"},
        )
        r = client.post("/search", data={"query": "ocean", "top_k": "3"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["count"] >= 1
        assert data["matches"][0]["note"] == "ocean at dusk"

    def test_conversational_nearby_query_uses_coordinates(self, client):
        memory_id = client.post(
            "/memory",
            files={"photo": ("img.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"note": "quiet corner", "tags": "cafe", "mood": "calm", "lat": "50.061", "lon": "19.938"},
        ).json()["memory_id"]

        data = client.post(
            "/chat",
            json={"message": "#cafe calm nearby", "lat": 50.06, "lon": 19.94},
        ).json()

        assert data["status"] == "success"
        assert data["memories"][0]["id"] == memory_id
        assert data["memories"][0]["distance_km"] is not None
        assert data["filters"] == {"tags": ["cafe"], "moods": ["calm"], "nearby": True}
        assert "closest first" in data["answer"]

    def test_shopping_want_phrase_is_a_status_not_a_tag(self, client):
        wanted_id = client.post(
            "/memory",
            files={"photo": ("want.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "shopping", "shopping_status": "want"},
        ).json()["memory_id"]
        client.post(
            "/memory",
            files={"photo": ("bought.jpg", _tiny_jpeg(), "image/jpeg")},
            data={"tags": "shopping", "shopping_status": "bought"},
        )

        data = client.post("/chat", json={"message": "#shopping want"}).json()

        assert [memory["id"] for memory in data["memories"]] == [wanted_id]
        assert data["filters"] == {"tags": ["shopping"], "shopping_status": "want"}


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


class TestInitDbMigration:
    def test_init_db_adds_missing_columns_without_data_loss(self, tmp_path, monkeypatch):
        old_db_path = str(tmp_path / "legacy_memories.db")
        old_images_dir = str(tmp_path / "legacy_images")
        monkeypatch.setattr(app_module, "DB_PATH", old_db_path)
        monkeypatch.setattr(app_module, "IMAGES_DIR", old_images_dir)

        conn = sqlite3.connect(old_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE memories (
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
            cursor.execute("""
                INSERT INTO memories (ts, lat, lon, image_path, image_emb, text_note, text_emb)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("2026-01-01T00:00:00", 1.0, 2.0, "images/legacy.jpg", b"img", "legacy note", b"text"))
            conn.commit()
        finally:
            conn.close()

        app_module.init_db()

        conn2 = sqlite3.connect(old_db_path)
        try:
            cursor2 = conn2.cursor()
            cursor2.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor2.fetchall()}
            assert {
                "user_note", "bepo_summary", "tags", "mood", "place_hint",
                "taken_at", "taken_at_source", "location_source",
                "note_created_at", "note_updated_at",
                "context_type", "shopping_status", "shopping_status_updated_at",
                "place_id",
            }.issubset(columns)

            cursor2.execute(
                "SELECT text_note, lat, lon, taken_at, note_created_at, note_updated_at, context_type "
                "FROM memories"
            )
            row = cursor2.fetchone()
            assert row == (
                "legacy note", 1.0, 2.0, None,
                "2026-01-01T00:00:00", "2026-01-01T00:00:00",
                "physical",
            )
            cursor2.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_status_history'")
            assert cursor2.fetchone() is not None
            cursor2.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'places'")
            assert cursor2.fetchone() is not None
        finally:
            conn2.close()
