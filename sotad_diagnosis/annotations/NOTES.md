# Annotation notes (`annotate_keyframes.py`)

Per-video observations made while annotating the accident area in the 90 test accidents. They complement
`keyframe_boxes.csv`, which only stores the box and the flag (ok / uncertain / not_visible).

## Protocol (fixed on 2026-10-06, during the annotation; before comparing with the motion region)

- One box per video, enclosing **the whole objects involved** (vehicles, motorcycles, people), not only the point of
  contact.
- Frame: the first one in which the accident becomes clear, starting from the keyframe (`dt_s` in the CSV).
- Object leaving the frame: only the visible part. Occluded part: probable extent if it is easy to infer; otherwise
  the visible part + `uncertain`.
- Very large vehicle with a small involvement: include the whole vehicle; huge box -> `uncertain`.
- Vehicles that only pass nearby are left out.
- The first 37 videos were annotated before this rule was written down.

## Per-video observations

| Video | Date | Flag in the CSV | Observation |
|---|---|---|---|
| 39 | 2026-10-06 | uncertain (corrected in the CSV) | Accident not visible: it is occluded. Box at the probable location. Corresponds to the "Occlusion" type of Fig. 6 of the SO-TAD paper |
