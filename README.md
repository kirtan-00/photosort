# PhotoSort

Local offline tool to index, search, and organize photos with face clustering and sharpness scoring.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
scripts/fetch_models.sh
```

## Launching

Double-click `PhotoSort.app` in Finder. It will prompt you to pick a folder, then open the browser to the sorting UI at http://localhost:7777.

## Commands

**Index a folder and extract face embeddings:**
```
python -m photosort.cli index ~/Pictures/Shot_001
python -m photosort.cli index ~/Pictures/Shot_001 --no-faces  # skip face detection (a later run with faces picks them up)
python -m photosort.cli index ~/Pictures/Shot_001 --retry-errors  # re-try photos that failed to decode last time
```

**Search photos by caption (MobileCLIP zero-shot):**
```
python -m photosort.cli find ~/Pictures/Shot_001 "a person smiling outdoors"
python -m photosort.cli find ~/Pictures/Shot_001 "close-up" --sharp 85 --limit 20
```

**Cluster faces and organize into person/group/solo folders:**
```
python -m photosort.cli people ~/Pictures/Shot_001
python -m photosort.cli people ~/Pictures/Shot_001 --eps 0.3  # stricter clustering
```

**Start the local web UI:**
```
python -m photosort.cli serve ~/Pictures/Shot_001 --open
```

**Run benchmarks:**
```
python -m photosort.cli bench ~/Pictures/Shot_001 --n 200
```

## Storage & Exports

The index and thumbnails live in `~/Library/Application Support/photosort/<shoot-slug>/`. Your source folder is never modified.

Exports (by default) copy files to `~/Desktop/photosort-out/<shoot>/<selection-name>/`. Use `--mode symlink` to create symlinks instead, or `--mode csv` to write a manifest.

## Notes

Sharpness is measured on the subject: if a face exists, it's scored on the eye region; otherwise, on the sharpest tiles in the frame. Shallow depth-of-field portraits won't be flagged as blurry.

Face clustering uses YuNet (detection) + SFace (embeddings) with DBSCAN. Tune `--eps` on your own shoot: lower values mean stricter clustering (fewer false matches). Start at 0.3-0.5 and adjust.

Search uses MobileCLIP-S1 (zero-shot, no LLM, fully local).
