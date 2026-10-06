# The 3-minute tour video

`website/docs/assets/video/visionserve-3min.mp4` (shown on the home page): 1280x720, 168 s, English
voice-over (Kokoro `am_michael`, Apache-2.0) and burned-in subtitles. Made with the
narrated-tutorial-video recipe: real outputs, a storyboard, offline TTS, then an automatic check
(Whisper recall per scene).

| file | what |
|---|---|
| `narration.json` | the spoken text, one entry per scene (numbers and acronyms spelled the way they are said) |
| `storyboard.json` | the scenes: terminals, code, diagrams, figures, tables |
| `out/*.txt` | outputs of the real runs, verbatim (some lines dropped, none edited) |
| `figures/*.png` | figures drawn from the real runs (`make_figures.py`; it reads the capture logs and photos of the run, which are not in the repo) |

Rebuild (needs `uv` and `ffmpeg`; the recipe installs Kokoro + Whisper CPU into `.video-env`):

```bash
cd website/video
<skill>/scripts/setup_env.sh .video-env
.video-env/bin/python <skill>/scripts/tts.py narration.json voice --voice am_michael --speed 1.3
.video-env/bin/python <skill>/scripts/render_video.py storyboard.json --out visionserve-3min.mp4
```

What was run for it: RF-DETR nano (official Roboflow checkpoint) converted with `visionserve convert`,
then `check`, `sensitivity`, `optimize`, `bench` on 15 photos (COCO samples and tabletop scenes); the container demo
ran the published `mtbui2010/visionserve:latest-cpu` on port 11436 (11435 was busy). The accuracy table is from an
earlier run on a fine-tuned RF-DETR small (215 held-out photos). Timings are for one shared machine.
The generated files `voice/`, `stills/`, `.video-env/` are not committed.
