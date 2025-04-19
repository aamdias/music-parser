# Lyrics and Chords Transcription API

This FastAPI application receives a music audio file (e.g., MP3), processes it end-to-end, and returns:
  - Timestamped song lyrics (via OpenAI Whisper with whisper-timestamped)
  - Timestamped chord events (via autochord)
  - Metadata: title, duration, BPM (optional), key (optional), source, and model versions

## Requirements
-- Python 3.8+
-- FFmpeg installed and available in your PATH
-- (Optional) GPU with CUDA for faster model inference
-- No external API keys required; uses local open-source models (whisper-timestamped, autochord)
-- Recommended: create and activate a Python virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
-- Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the app:
   ```bash
   uvicorn main:app --reload
   ```
2. Send a POST request to `/analyze` with form-data:
   - Key: `file`
   - Value: audio file to process
3. Receive a JSON response matching the schema in [`SPEC.md`](SPEC.md).

## JSON Response Format
```jsonc
{
  "metadata": {
    "title": "filename.mp3",
    "artist": "",
    "duration_sec": 233.72,
    "bpm": 97,
    "key": "",
    "source": "filename.mp3",
    "model_versions": {
      "lyrics": "whisper-large-v3-timestamped",
      "chords": "1.3.0"
    }
  },
  "lyrics": [
    { "start": 12.53, "end": 16.42, "text": "When I find myself in times of trouble" },
    ...
  ],
  "chords": [
    { "start": 12.53, "end": 14.31, "label": "C:maj" },
    ...
  ]
}
```

## Troubleshooting

- If you encounter errors building or installing `whisper-timestamped` due to `openai-whisper`:
  ```bash
  pip install --upgrade pip setuptools wheel
  ```
-- If the build still fails on Python 3.13 (KeyError '__version__'), pin `openai-whisper` to a known working version (e.g., 20230314):
  ```bash
  pip install openai-whisper==20230314
  pip install -r requirements.txt
  ```
  Alternatively, consider using Python 3.11 or 3.10, as some versions of `openai-whisper` may not build cleanly on Python 3.13.
```