# Bepo GUI

A minimal desktop app for Bepo — drag-and-drop photos, add notes and GPS
coordinates, and search your memories, all without leaving your desktop.

## Prerequisites

Install the backend dependencies (from the repo root):

```bash
pip install -r requirements.txt
```

Then install the GUI-specific dependencies:

```bash
pip install -r gui/requirements_gui.txt
```

> **Note:** `tkinter` ships with the standard Python installer on Windows and
> macOS. On Linux you may need to install it separately, e.g.
> `sudo apt install python3-tk`.

## Running

Both the backend and the GUI must run at the same time.

**1. Start the backend** (from the repo root):

```bash
python main.py
```

This starts the FastAPI server on `http://127.0.0.1:8000`.

**2. Start the GUI** (in a second terminal, from the repo root):

```bash
python gui/app.py
```

The status bar at the bottom of the window turns green when the GUI has
successfully connected to the backend. If it shows red, make sure `main.py`
is running first.

## Usage

### Add Memory tab
- Drop an image onto the dashed area **or** click **Browse…** to pick one.
- Optionally fill in a **Note**, **Lat**, and **Lon**.
- Click **Save Memory** — the response area shows the saved memory ID.

### Search tab
- Type a query and press **Enter** or click **Search**.
- The best-matching memory is shown with its note, score, timestamp, and a
  thumbnail of the original photo.
