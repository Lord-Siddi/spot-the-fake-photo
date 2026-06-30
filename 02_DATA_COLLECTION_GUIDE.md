# Data Collection Guide (for you, not the agent)

The agent building this project cannot take photos with a phone, so this one step needs
you. It takes about 30–60 minutes, exactly as the assignment says. Do this before asking
Antigravity to run `train.py`.

## What to capture

### `data/real/` — about 50 photos of **real things** (not on a screen)
Vary these as much as you can, since the held-out test set will be different photos than
yours and your model needs to generalize:
- Indoor and outdoor
- Different lighting (daylight, lamp light, dim room, direct sun, shade)
- Different subjects (objects, food, plants, people, rooms, text/documents on paper)
- Different distances and angles (close-up, far, tilted)
- A few photos of printed documents/photos held in your hand counts as "real" only if
  it's a paper printout you photographed directly — but note the assignment also lists
  printouts under the "screen" category in one place, so to be safe: **put any printout
  re-photographing another image into `screen/`, and keep `real/` for genuine first-generation
  photos of physical objects/scenes only.**

### `data/screen/` — about 50 photos of **a screen or printout showing a picture**
- Take photos of a photo/image displayed on: a phone screen, a laptop/monitor screen, a
  TV, and a printed photo or printout, if you can get a few of each
- Vary angle (straight-on and at a slight angle), distance, and screen brightness
- Try both with and without visible glare/reflection
- Try at least a couple of different screens/devices (e.g. your phone + a laptop) — this
  matters because the judge's held-out set will use different screens than yours, and a
  model trained on only one screen type will overfit to that screen's specific moiré pattern

## Folder structure

```
data/
├── real/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── screen/
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

Any common image format (jpg/png) is fine. File names don't matter — only which folder
they're in.

## Tips for a stronger dataset

- More variety beats more volume: 50 varied photos generalize better than 100 similar ones.
- Don't worry about getting it "perfectly balanced" beyond roughly 50/50 — a small
  imbalance is fine and the agent's training code will handle it.
- Once both folders have photos in them, tell Antigravity to proceed with `train.py`.
