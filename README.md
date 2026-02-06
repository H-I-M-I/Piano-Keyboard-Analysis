# IACV-Project

## Extract frames (concise)

Run the most concise extraction command (from the project root, inside the `iacvProject` venv):

```bash
python src/frames_extractor.py piano_video_2
```

This will read `data/raw_videos/piano_video_2.mp4` and write the final cleaned frames to `data/frames/piano_video_2/cleaned`.

## Frames extractor arguments

- `video_name` (positional): video filename without path or extension (default: `piano_video_1`)
- `--fps`: frames per second to sample (default: `10`)
- `--maxframes`: maximum number of sampled frames to consider (default: `600`)
- `--deviation`: similarity deviation threshold used for initial filtering (default: `1`)
- `--show`: show a sample extracted frame (opens a plot)
- `--save`: save intermediate sampled frames in addition to the final cleaned frames

## Full example (explicit arguments)

```bash
python src/frames_extractor.py piano_video_2 --fps 10 --maxframes 600 --deviation 1 --show --save
```

Notes:
- Use double backslashes or wrap paths in quotes on Windows when specifying paths elsewhere.
- If `python` on your PATH is not the project venv, run the venv Python directly: `iacvProject\Scripts\python.exe src/frames_extractor.py ...`
