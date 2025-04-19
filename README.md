# Lyrics and Chords Transcription API

This FastAPI application receives a music audio file (e.g., MP3), processes it end-to-end, and returns:
  - Timestamped song lyrics (via OpenAI GPT-4o transcribe model)
  - Timestamped chord events (**Upcoming: AI-powered chord recognition**)
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
   python3 main.py
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
      "lyrics": "gpt-4o-transcribe",
      "chords": "1.3.0 (to be replaced by AI model)"
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

## Project Roadmap

**Note:** The current chord detection uses `autochord`, which is not accurate enough for professional or complex music. We are actively developing a new AI-based chord recognition system leveraging deep learning and the latest research. This will significantly improve chord accuracy and robustness.

### Planned Improvements
- Replace `autochord` with a custom-trained neural network for chord recognition
- Better support for complex chords and polyphony
- Improved handling of noisy or live recordings
- Open to community contributions for model training, evaluation, and dataset gathering

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