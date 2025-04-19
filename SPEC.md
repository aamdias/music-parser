High Livel Goal
This app should be able to receive an music audio file and generates an structured output of the song lyrics with song chords with timestamps for each lyrics and for each chord.

1 · End‑to‑end pipeline
#	Stage	What happens	Suggested tooling
1	Ingest	Receive/stream the .mp3. Convert to 16‑kHz mono WAV to keep every lib happy.	ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav
2	(Optional) source separation	Strip accompaniment to make speech & chords cleaner.	demucs --two-stems vocals out.wav
3	Lyrics + timestamps	• Run OpenAI Whisper (or local whisper-large‑v3)	
• For word‑level timing use the whisper‑timestamped patch	pip install whisper-timestamped – returns every word with {start, end, text} 
GitHub
4	Chord recognition	Feed the original (or mixture) audio to a chord‑transcription model. Two good OSS options:	
  • autochord – Bi‑LSTM‑CRF, 25 chord classes, 1‑line API call 
GitHub
  • chord‑extractor – wrapper around Chordino + parallel batch support 
Reddit
5	Beat/BPM (optional)	librosa.beat.beat_track if you want bar‑aligned chords	
6	Alignment / merge	Keep the two result lists separate (simpler) or snap chords to the nearest lyric line start.	
7	Export	Emit the JSON structure below (deterministic → easy to parse in web / mobile / LLAMA).	
2 · Predictable JSON output
jsonc
Copiar código
{
  "metadata": {
    "title": "string",
    "artist": "string",
    "duration_sec": 233.72,
    "bpm": 97,
    "key": "A♭ major",
    "source": "path‑or‑url/of/file.mp3",
    "model_versions": {
      "lyrics": "whisper-large-v3-timestamped",
      "chords": "autochord‑1.3.0"
    }
  },

  // ONE entry per sung line ----------------------------------------------
  "lyrics": [
    {
      "start": 12.53,          // seconds
      "end":   16.42,
      "text": "When I find myself in times of trouble"
    },
    ...
  ],

  // ONE entry per harmonic event -----------------------------------------
  "chords": [
    { "start": 12.53, "end": 14.31, "label": "C:maj" },
    { "start": 14.31, "end": 16.42, "label": "G:maj" },
    ...
  ]
}
Why two parallel arrays instead of interleaving chords inside each lyric?

* It mirrors how the tools spit out data → zero post‑processing headaches
* UI‑side you can still overlay the two timelines easily
* If you later want bar‑level resolution you won’t have to reopen the schema.

3 · Implementation notes & gotchas
Model choice trade‑offs

whisper‑large‑v3 already understands sung speech surprisingly well. If the song has heavy reverb or backing vocals, run Demucs first.

autochord.recognize() returns a [(start, end, label)] list in one call; accuracy ≈ 67 % on the Ismir‑2021 set. For pop/rock that’s usually “good enough to put chords above lyrics”, but you can ensemble with Chordino for jazz extensions.

Performance – Both models are GPU‑friendly; for batch jobs use a single RTX 4090 or A100. Real‑time inference on a 4‑minute track ≈ 25‑30 s total (8‑10 s Whisper, 15‑20 s autochord).

Language coverage – Whisper handles >90 languages; whisper‑timestamped keeps word‑level offsets even when the lyrics mix languages.

Post‑processing tips

Merge adjacent identical chords so you don’t flood the array with micro‑events.

Clamp chord changes to detected beats if you need tidy bar diagrams.

Run a spell‑checker on lyrics; sung words sometimes slur.

Versioning for reproducibility – Put model name + git SHA in metadata.model_versions (see schema). Your downstream apps can then refuse old files or auto‑migrate.

Future‑proofing – If you later want MIDI, add a midi key with a Base‑64 blob of the track or a URL to a .mid file.

TL;DR
Use Whisper + whisper‑timestamped for line‑accurate lyrics, autochord (or chord‑extractor) for chords, keep them in two timestamped arrays inside a single JSON object.
That structure is minimal, deterministic, and extensible—perfect for an AI agent or a front‑end player.