#!/usr/bin/env python3
"""
Bepo GUI — minimal desktop app wrapping the Bepo FastAPI backend.
Run with: python gui/app.py  (backend must already be running on port 8000)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import requests
except ImportError:
    requests = None  # handled at runtime with a friendly message

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

# ── constants ────────────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8000"

# colour palette — dark neutral
BG       = "#1e1e1e"
SURFACE  = "#2a2a2a"
BORDER   = "#3a3a3a"
FG       = "#e8e8e8"
FG_DIM   = "#888888"
ACCENT   = "#4f8ef7"
ACCENT_H = "#6aa3ff"   # hover
DANGER   = "#e05c5c"
SUCCESS  = "#4caf8e"

# typography
FONT_UI   = ("Segoe UI", 10) if sys.platform == "win32" else ("Arial", 10)
FONT_SM   = ("Segoe UI", 9)  if sys.platform == "win32" else ("Arial", 9)
FONT_H1   = ("Segoe UI", 13, "bold") if sys.platform == "win32" else ("Arial", 13, "bold")
FONT_CODE = ("Consolas", 9) if sys.platform == "win32" else ("Courier", 9)

THUMB_SIZE = (120, 120)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_thumb(path: str) -> "ImageTk.PhotoImage | None":
    """Return a thumbnail PhotoImage for *path*, or None on failure."""
    if not (Image and ImageTk):
        return None
    try:
        img = Image.open(path)
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _styled_button(parent, text, command, accent=True, **kw):
    bg = ACCENT if accent else SURFACE
    fg = "#ffffff" if accent else FG
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=ACCENT_H if accent else BORDER,
        activeforeground="#ffffff" if accent else FG,
        relief="flat",
        bd=0,
        padx=14,
        pady=6,
        font=FONT_UI,
        cursor="hand2",
        **kw,
    )
    return btn


def _styled_entry(parent, **kw):
    e = tk.Entry(
        parent,
        bg=SURFACE,
        fg=FG,
        insertbackground=FG,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        font=FONT_UI,
        **kw,
    )
    return e


def _styled_text(parent, height=4, **kw):
    t = tk.Text(
        parent,
        bg=SURFACE,
        fg=FG,
        insertbackground=FG,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        font=FONT_CODE,
        height=height,
        wrap="word",
        state="disabled",
        **kw,
    )
    return t


def _text_set(widget: tk.Text, text: str):
    widget.config(state="normal")
    widget.delete("1.0", "end")
    widget.insert("end", text)
    widget.config(state="disabled")


# ── main application ──────────────────────────────────────────────────────────


class BepoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bepo")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(560, 640)

        # state
        self._add_image_path: str | None = None
        self._add_thumb: "ImageTk.PhotoImage | None" = None
        self._search_thumb: "ImageTk.PhotoImage | None" = None

        self._build_ui()
        self._check_backend_async()

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # header
        hdr = tk.Frame(self.root, bg=BG, pady=14)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="Bepo", font=FONT_H1, bg=BG, fg=FG).pack(side="left")
        tk.Label(
            hdr,
            text="memory, searchable",
            font=FONT_SM,
            bg=BG,
            fg=FG_DIM,
        ).pack(side="left", padx=(8, 0), pady=(3, 0))

        # separator
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # notebook
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Bepo.TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            "Bepo.TNotebook.Tab",
            background=SURFACE,
            foreground=FG_DIM,
            padding=[16, 8],
            font=FONT_UI,
            borderwidth=0,
        )
        style.map(
            "Bepo.TNotebook.Tab",
            background=[("selected", BG)],
            foreground=[("selected", FG)],
        )

        nb = ttk.Notebook(self.root, style="Bepo.TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        add_frame = tk.Frame(nb, bg=BG)
        search_frame = tk.Frame(nb, bg=BG)

        nb.add(add_frame, text="  Add Memory  ")
        nb.add(search_frame, text="  Search  ")

        self._build_add_tab(add_frame)
        self._build_search_tab(search_frame)

        # status bar
        sb = tk.Frame(self.root, bg=SURFACE, height=28)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        self._status_dot = tk.Label(sb, text="●", font=FONT_SM, bg=SURFACE, fg=FG_DIM)
        self._status_dot.pack(side="left", padx=(12, 4), pady=6)
        self._status_label = tk.Label(
            sb, text="Checking backend…", font=FONT_SM, bg=SURFACE, fg=FG_DIM
        )
        self._status_label.pack(side="left")

    def _build_add_tab(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        # drop zone
        tk.Label(wrap, text="Photo", font=FONT_SM, bg=BG, fg=FG_DIM).pack(anchor="w")

        drop_outer = tk.Frame(wrap, bg=BORDER, bd=0)
        drop_outer.pack(fill="x", pady=(4, 0))

        self._drop_zone = tk.Frame(drop_outer, bg=SURFACE, cursor="hand2")
        self._drop_zone.pack(fill="x", padx=1, pady=1)

        # inner content of drop zone
        self._drop_inner = tk.Frame(self._drop_zone, bg=SURFACE)
        self._drop_inner.pack(fill="x", padx=16, pady=20)

        self._drop_hint = tk.Label(
            self._drop_inner,
            text="Drop image here",
            font=FONT_SM,
            bg=SURFACE,
            fg=FG_DIM,
        )
        self._drop_hint.pack()

        self._add_thumb_label = tk.Label(self._drop_inner, bg=SURFACE)
        self._add_path_label = tk.Label(
            self._drop_inner,
            text="",
            font=FONT_SM,
            bg=SURFACE,
            fg=FG_DIM,
            wraplength=380,
        )

        browse_row = tk.Frame(wrap, bg=BG)
        browse_row.pack(fill="x", pady=(8, 16))
        _styled_button(browse_row, "Browse…", self._browse_image, accent=False).pack(
            side="left"
        )

        # note
        tk.Label(wrap, text="Note  (optional)", font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            anchor="w"
        )
        self._note_entry = _styled_entry(wrap)
        self._note_entry.pack(fill="x", pady=(4, 14))

        # lat / lon
        coords_row = tk.Frame(wrap, bg=BG)
        coords_row.pack(fill="x", pady=(0, 14))
        coords_row.columnconfigure(0, weight=1)
        coords_row.columnconfigure(1, weight=1)

        lat_col = tk.Frame(coords_row, bg=BG)
        lat_col.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        tk.Label(lat_col, text="Lat  (optional)", font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            anchor="w"
        )
        self._lat_entry = _styled_entry(lat_col, width=12)
        self._lat_entry.pack(fill="x", pady=(4, 0))

        lon_col = tk.Frame(coords_row, bg=BG)
        lon_col.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        tk.Label(lon_col, text="Lon  (optional)", font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            anchor="w"
        )
        self._lon_entry = _styled_entry(lon_col, width=12)
        self._lon_entry.pack(fill="x", pady=(4, 0))

        # save button
        self._save_btn = _styled_button(wrap, "Save Memory", self._save_memory)
        self._save_btn.pack(anchor="w", pady=(4, 14))

        # response area
        tk.Label(wrap, text="Response", font=FONT_SM, bg=BG, fg=FG_DIM).pack(anchor="w")
        self._add_response = _styled_text(wrap, height=4)
        self._add_response.pack(fill="x", pady=(4, 0))

        # wire up drag-and-drop
        if _DND_AVAILABLE:
            for widget in (self._drop_zone, self._drop_inner, self._drop_hint):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self._drop_hint.config(text="Drop not available — use Browse")

    def _build_search_tab(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(wrap, text="Search query", font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            anchor="w"
        )
        q_row = tk.Frame(wrap, bg=BG)
        q_row.pack(fill="x", pady=(4, 16))

        self._query_entry = _styled_entry(q_row)
        self._query_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._query_entry.bind("<Return>", lambda _e: self._do_search())

        _styled_button(q_row, "Search", self._do_search).pack(side="left")

        # result label
        tk.Label(wrap, text="Result", font=FONT_SM, bg=BG, fg=FG_DIM).pack(anchor="w")

        # result card
        self._result_card = tk.Frame(wrap, bg=SURFACE, bd=0)
        self._result_card.pack(fill="x", pady=(4, 14))

        result_inner = tk.Frame(self._result_card, bg=SURFACE, padx=14, pady=14)
        result_inner.pack(fill="x")

        self._search_thumb_label = tk.Label(result_inner, bg=SURFACE)
        self._search_thumb_label.pack(side="left", anchor="n", padx=(0, 14))

        info_col = tk.Frame(result_inner, bg=SURFACE)
        info_col.pack(side="left", fill="both", expand=True)

        self._res_note = tk.Label(
            info_col,
            text="No results yet.",
            font=FONT_UI,
            bg=SURFACE,
            fg=FG,
            wraplength=320,
            justify="left",
            anchor="w",
        )
        self._res_note.pack(anchor="w")

        self._res_meta = tk.Label(
            info_col,
            text="",
            font=FONT_SM,
            bg=SURFACE,
            fg=FG_DIM,
            wraplength=320,
            justify="left",
            anchor="w",
        )
        self._res_meta.pack(anchor="w", pady=(4, 0))

        # raw response
        tk.Label(wrap, text="Raw response", font=FONT_SM, bg=BG, fg=FG_DIM).pack(
            anchor="w"
        )
        self._search_response = _styled_text(wrap, height=6)
        self._search_response.pack(fill="x", pady=(4, 0))

    # ── image selection / drop ───────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.gif *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self._set_add_image(path)

    def _on_drop(self, event):
        raw = event.data.strip()
        # tkinterdnd2 wraps paths with braces when there are spaces
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0]  # take first file if multiple dropped
        self._set_add_image(path)

    def _set_add_image(self, path: str):
        self._add_image_path = path
        self._drop_hint.pack_forget()
        self._add_thumb_label.config(image="")
        self._add_path_label.pack_forget()

        thumb = _make_thumb(path)
        if thumb:
            self._add_thumb = thumb  # keep reference
            self._add_thumb_label.config(image=thumb)
            self._add_thumb_label.pack(pady=(0, 6))
        self._add_path_label.config(text=os.path.basename(path))
        self._add_path_label.pack()

    # ── API calls ────────────────────────────────────────────────────────────

    def _check_backend_async(self):
        threading.Thread(target=self._check_backend, daemon=True).start()

    def _check_backend(self):
        if requests is None:
            self._set_status(False, "requests not installed — pip install requests")
            return
        try:
            r = requests.get(BASE_URL + "/", timeout=3)
            if r.status_code == 200:
                self._set_status(True, "Backend connected")
            else:
                self._set_status(False, f"Backend returned {r.status_code}")
        except Exception:
            self._set_status(
                False, "Backend not running — start with: python main.py"
            )

    def _set_status(self, ok: bool, msg: str):
        color = SUCCESS if ok else DANGER
        label_fg = color if not ok else FG_DIM

        def _apply():
            self._status_dot.config(fg=color)
            self._status_label.config(text=msg, fg=label_fg)

        self.root.after(0, _apply)

    def _save_memory(self):
        if requests is None:
            _text_set(self._add_response, "Error: requests library not installed.")
            return
        if not self._add_image_path:
            _text_set(self._add_response, "Please select an image first.")
            return

        self._save_btn.config(state="disabled", text="Saving…")
        _text_set(self._add_response, "Sending…")

        note = self._note_entry.get().strip() or None
        lat_raw = self._lat_entry.get().strip()
        lon_raw = self._lon_entry.get().strip()
        lat = float(lat_raw) if lat_raw else None
        lon = float(lon_raw) if lon_raw else None
        image_path = self._add_image_path

        def _work():
            try:
                with open(image_path, "rb") as f:
                    files = {"photo": (os.path.basename(image_path), f)}
                    data = {}
                    if note:
                        data["note"] = note
                    if lat is not None:
                        data["lat"] = lat
                    if lon is not None:
                        data["lon"] = lon
                    r = requests.post(BASE_URL + "/memory", files=files, data=data, timeout=30)
                result = r.json()
                mid = result.get("memory_id", "?")
                text = (
                    f"Memory saved!  (id: {mid})\n\n"
                    + _pretty_json(result)
                )
            except requests.exceptions.ConnectionError:
                text = "Error: Cannot reach backend.\nMake sure it's running: python main.py"
            except Exception as exc:
                text = f"Error: {exc}"
            self.root.after(0, lambda: self._save_done(text))

        threading.Thread(target=_work, daemon=True).start()

    def _save_done(self, text: str):
        _text_set(self._add_response, text)
        self._save_btn.config(state="normal", text="Save Memory")
        self._check_backend_async()

    def _do_search(self):
        if requests is None:
            _text_set(self._search_response, "Error: requests library not installed.")
            return
        query = self._query_entry.get().strip()
        if not query:
            return

        _text_set(self._search_response, "Searching…")
        self._res_note.config(text="Searching…")
        self._res_meta.config(text="")

        def _work():
            try:
                r = requests.post(BASE_URL + "/search", data={"query": query}, timeout=15)
                result = r.json()
            except requests.exceptions.ConnectionError:
                result = None
                err = "Error: Cannot reach backend.\nMake sure it's running: python main.py"

                def _show_conn_err(e=err):
                    _text_set(self._search_response, e)
                    self._res_note.config(text="Backend unreachable.")

                self.root.after(0, _show_conn_err)
                return
            except Exception as exc:
                result = None
                err_msg = f"Error: {exc}"

                def _show_exc(m=err_msg):
                    _text_set(self._search_response, m)
                    self._res_note.config(text="An error occurred.")

                self.root.after(0, _show_exc)
                return
            self.root.after(0, lambda: self._search_done(result))

        threading.Thread(target=_work, daemon=True).start()

    def _search_done(self, result: dict):
        _text_set(self._search_response, _pretty_json(result))

        if result.get("status") == "success" and "match" in result:
            match = result["match"]
            note_text = match.get("note") or "(no note)"
            score = match.get("score", 0)
            ts = match.get("timestamp", "")
            lat = match.get("lat")
            lon = match.get("lon")

            self._res_note.config(text=note_text)
            meta_parts = [f"score: {score:.4f}", f"saved: {ts[:19] if len(ts) >= 19 else ts}"]
            if lat is not None and lon is not None:
                meta_parts.append(f"({lat}, {lon})")
            self._res_meta.config(text="  ·  ".join(meta_parts))

            # thumbnail from local image_path
            img_path = match.get("image_path", "")
            thumb = _make_thumb(img_path)
            if thumb:
                self._search_thumb = thumb
                self._search_thumb_label.config(image=thumb)
            else:
                self._search_thumb_label.config(image="")
        elif result.get("status") == "no_results":
            self._res_note.config(text="No memories in database yet.")
            self._res_meta.config(text="")
            self._search_thumb_label.config(image="")
        else:
            self._res_note.config(text="No match found.")
            self._res_meta.config(text="")
            self._search_thumb_label.config(image="")

        self._check_backend_async()


# ── utilities ─────────────────────────────────────────────────────────────────


def _pretty_json(obj) -> str:
    import json
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


# ── entry point ───────────────────────────────────────────────────────────────


def main():
    if _DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.geometry("620x720")
    app = BepoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
