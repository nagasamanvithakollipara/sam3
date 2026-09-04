#!/usr/bin/env python3
"""
Run SAM3 on 360 equirectangular frames by reprojecting each frame into several
perspective views, detecting in each view, and mapping the masks back onto the
original equirectangular image.

Detecting directly on an equirectangular frame fails for anything near the poles:
the projection smears it across the bottom/top edge and SAM3 scores it at ~0.
Reprojecting to an oblique perspective view removes that distortion.

Supports the same local-folder / s3:// inputs as batch_process_s3_images.py.

Performance notes
-----------------
The naive version of this script ran SAM3 once per view, i.e. 9 full backbone
passes per frame, and did all mask bookkeeping at full equirectangular
resolution (16 MP for a 5760x2880 frame). Both are avoided here:

* All views of a frame go through the backbone as ONE batch, and every
  (view, concept) pair is a single query row in ONE grounding pass. SAM3's
  ``forward_grounding`` is natively batched over ``img_ids``/``text_ids``, so
  this is the same computation with the per-call overhead paid once.
* Masks are back-projected onto a decimated equirect grid (``--mask_scale``)
  and only upsampled when the final mask/overlay is written. Dedupe, area and
  overlap maths then run on ~1/16th of the pixels.
* Frame fetch/decode and output encoding happen on their own threads, one frame
  either side of the GPU, so ~0.2s of per-frame I/O costs nothing.
* ``--model_resolution`` actually works now: the ViT bakes its RoPE tables at
  build time, so anything but 1008 used to assert. See
  ``retune_backbone_resolution``. This is the dial that buys real time.

Steady-state cost on a T4 (shared), 9 views, 4096x2048 frames: 6.8s/frame at the
native 1008px, 1.6s at 448, 0.7s at 336 (``--fast``). The GPU forward pass is
~95% of that, so anything below 1s/frame is bought with resolution or with fewer
views, not with more plumbing.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

SAM3_PATH = os.getenv('SAM3_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam3'))
if os.path.exists(SAM3_PATH):
    sys.path.insert(0, SAM3_PATH)

from s3_detection_segmentation import SAM3Detector, S3ImageProcessor
from batch_process_s3_images import list_local_images, list_s3_images

# yaw, pitch pairs covering the nadir (where the operator's own helmet sits) plus
# the horizon ring. Pitch -60 is where helmets score best; see the sweep in README.
DEFAULT_VIEWS = [
    (0, -60), (90, -60), (180, -60), (270, -60),   # oblique down: catches the rig + nearby workers
    (0, 0), (90, 0), (180, 0), (270, 0),           # horizon: workers at distance
    (0, -90),                                      # nadir: the operator's own helmet, in every frame
]

# Second-chance views, used only on frames where the pass above found nothing.
# The default ring puts a view seam every 90 deg; an object straddling one is
# clipped in every view it appears in and scores below box_threshold. These yaws
# are offset by 45 deg so a seam-clipped object lands mid-view, plus a true nadir
# view for the rig and anyone right beside it.
RESCUE_VIEWS = [
    (45, -60), (135, -60), (225, -60), (315, -60),
    (45, 0), (135, 0), (225, 0), (315, 0),
]


def _rotation(yaw_deg, pitch_deg):
    p = np.radians(pitch_deg)
    yw = np.radians(yaw_deg)
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(p), -np.sin(p)],
                      [0, np.sin(p), np.cos(p)]], dtype=np.float32)
    rot_y = np.array([[np.cos(yw), 0, np.sin(yw)],
                      [0, 1, 0],
                      [-np.sin(yw), 0, np.cos(yw)]], dtype=np.float32)
    return rot_x, rot_y


def build_view_maps(eq_h, eq_w, yaw_deg, pitch_deg, fov_deg, out_size):
    """Return (map_x, map_y) remapping a perspective view to equirect pixel coords."""
    f = 0.5 * out_size / np.tan(np.radians(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(out_size, dtype=np.float32),
                       np.arange(out_size, dtype=np.float32))
    x = (j - out_size / 2.0) / f
    y = (i - out_size / 2.0) / f
    z = np.ones_like(x)

    v = np.stack([x, y, z], axis=-1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)

    rot_x, rot_y = _rotation(yaw_deg, pitch_deg)
    v = v @ rot_x.T @ rot_y.T

    lon = np.arctan2(v[..., 0], v[..., 2])
    lat = np.arcsin(np.clip(v[..., 1], -1.0, 1.0))

    map_x = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.float32)
    map_y = ((lat / np.pi + 0.5) * eq_h).astype(np.float32)
    return map_x, map_y


def render_view(eq_image, map_x, map_y):
    return cv2.remap(eq_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def build_inverse_maps(eq_h, eq_w, yaw_deg, pitch_deg, fov_deg, out_size):
    """Return (inv_x, inv_y, valid): for each equirect pixel, its coords in the view.

    This is the inverse of build_view_maps. Scattering view pixels onto the
    equirect leaves holes wherever the equirect is denser than the view (badly so
    near the poles), so we instead sample the view for every equirect pixel.

    The projection is resolution-independent, so eq_h/eq_w here are the decimated
    mask-grid dimensions and out_size is the size of the view-space mask raster -
    neither has to match the resolution the image was actually rendered at.
    """
    lon = (np.arange(eq_w, dtype=np.float32) / eq_w - 0.5) * (2 * np.pi)
    lat = (np.arange(eq_h, dtype=np.float32) / eq_h - 0.5) * np.pi
    lon, lat = np.meshgrid(lon, lat)

    cos_lat = np.cos(lat)
    v = np.stack([cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)], axis=-1)

    rot_x, rot_y = _rotation(yaw_deg, pitch_deg)
    # forward was v @ rot_x.T @ rot_y.T, so invert in reverse order
    v = v @ rot_y @ rot_x

    f = 0.5 * out_size / np.tan(np.radians(fov_deg) / 2.0)
    z = v[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_x = v[..., 0] / z * f + out_size / 2.0
        inv_y = v[..., 1] / z * f + out_size / 2.0

    valid = (z > 1e-6) & (inv_x >= 0) & (inv_x < out_size) & (inv_y >= 0) & (inv_y < out_size)
    inv_x = np.where(valid, inv_x, -1).astype(np.float32)
    inv_y = np.where(valid, inv_y, -1).astype(np.float32)
    return inv_x, inv_y, valid


def mask_to_equirect(view_mask, inv_x, inv_y, valid):
    """Sample a view-space mask for every equirect pixel that the view covers."""
    src = view_mask.astype(np.uint8) * 255
    out = cv2.remap(src, inv_x, inv_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = (out > 127) & valid
    if not out.any():
        return None
    return out


def latitude_weights(eq_h, eq_w):
    """Per-row solid-angle weight for an equirect image.

    An equirect row near the pole covers the same pixel count as a row at the
    equator but a far smaller solid angle, so raw pixel areas there describe the
    projection rather than the object: a helmet at the nadir and the person
    wearing it both balloon to ~400-700k px and look like the same size. Weighting
    each row by cos(latitude) restores true relative size.
    """
    lat = (np.arange(eq_h, dtype=np.float32) / eq_h - 0.5) * np.pi
    return np.cos(lat).astype(np.float32)[:, None]


def mask_area(mask, weights=None):
    """Solid-angle area of a mask (falls back to a raw pixel count)."""
    if weights is None:
        return float(mask.sum())
    # Summing the per-row weights over each row's true count is far cheaper than
    # materialising a float array the size of the mask.
    return float(mask.sum(axis=1, dtype=np.int64) @ weights[:, 0])


def mask_overlap(a, b, weights=None):
    """Containment-aware overlap: intersection over the SMALLER mask.

    Views disagree on extent - one detection covers just the helmet, another the
    helmet plus the vest below it. Plain IoU scores those nested masks low and
    leaves both in the output, so we measure containment instead.
    """
    both = np.logical_and(a, b)
    if not both.any():
        return 0.0
    inter = mask_area(both, weights)
    return inter / min(mask_area(a, weights), mask_area(b, weights))


def _bbox_overlaps(a, b):
    """Cheap reject: do two detections' bounding boxes touch at all?"""
    if a is None or b is None:
        return False
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def dedupe(detections, iou_threshold, max_size_ratio=3.0, weights=None):
    """Same object seen in two overlapping views -> union their masks into one.

    Overlap alone is not enough to call two masks the same object. A helmet sits
    entirely inside the 'person' wearing it, and near the nadir the person mask
    smears across the full width of the equirect, so containment against it is
    ~1.0 for anything down there. Scored by confidence alone, that smear wins and
    deletes the precise helmet mask from every view.

    So two detections merge only if they also have comparable extent: the larger
    mask may be at most max_size_ratio times the smaller. Duplicate helmets from
    overlapping views are near-identical in size and still collapse; a helmet
    nested in a person does not.

    Merging is a union, not a pick. Every view sees only the part of an object
    inside its own frustum, so a person under the rig comes back as a top-down
    fragment from the nadir view and side fragments from the oblique ones. Keeping
    only the highest-scoring fragment discards the rest and leaves the object
    partly unsegmented; OR-ing the fragments together reconstructs it. A merged
    detection keeps the winner's score and phrase and records every contributing
    view in 'view'.

    Because merging grows a mask, the pass repeats until nothing changes: a
    fragment that failed the size test against a small early mask can pass once
    that mask has absorbed its neighbours.

    Areas and bounding boxes are computed once per detection rather than once per
    comparison, and a disjoint pair of boxes skips the mask intersection entirely.
    """
    debug = os.getenv('DEDUPE_DEBUG')
    for det in detections:
        if 'bbox' not in det:
            det['bbox'] = bbox_from_mask(det['mask'])
        if '_area' not in det:
            det['_area'] = mask_area(det['mask'], weights)

    def _refresh(d):
        d['bbox'] = bbox_from_mask(d['mask'])
        d['_area'] = mask_area(d['mask'], weights)

    def _match(det, k):
        """Do these two detections describe the same object?"""
        # The size test exists to stop a helmet being absorbed by the person
        # wearing it - a nesting that only happens between *different* concepts.
        # Two fragments of one object carry the same phrase, and their sizes are
        # set by how much of it each frustum happened to catch (a nadir view sees
        # a torso, an oblique view a pair of legs), so a size ratio says nothing
        # about whether they are the same thing. Applying the guard there is what
        # left objects partly unsegmented, so same-phrase pairs skip it and are
        # judged on containment alone.
        if det['phrase'] != k['phrase']:
            ratio = max(det['_area'], k['_area']) / max(min(det['_area'], k['_area']), 1e-6)
            # The size test does not depend on the masks, so apply it (and the
            # bbox test) before paying for a full-frame intersection.
            if ratio > max_size_ratio:
                return False
        if not _bbox_overlaps(det['bbox'], k['bbox']):
            return False
        return mask_overlap(det['mask'], k['mask'], weights) > iou_threshold

    kept = []
    for det in sorted(detections, key=lambda d: -d['score']):
        if debug:
            print(f"  cand {det['score']:.3f} {det['phrase']!r:16} {det['view']:10} "
                  f"area={det['_area']}")
        target = next((k for k in kept if _match(det, k)), None)
        if target is None:
            det['views'] = [det['view']]
            kept.append(det)
            continue
        if debug:
            print(f"       -> union into {target['phrase']!r} [{target['view']}]")
        target['mask'] = target['mask'] | det['mask']
        target['views'].append(det['view'])
        target['view'] = '+'.join(target['views'])
        _refresh(target)

    # A mask that grew by absorbing fragments may now match a detection it was
    # too small to match on the first pass, so keep folding until it settles.
    changed = True
    while changed and len(kept) > 1:
        changed = False
        for i, k in enumerate(kept):
            # Index, not the dict: list.remove/index compare with ==, and these
            # dicts hold numpy masks, so == returns an array and raises
            # "truth value of an array ... is ambiguous".
            j = next((n for n in range(i + 1, len(kept)) if _match(k, kept[n])), None)
            if j is None:
                continue
            other = kept[j]
            if debug:
                print(f"  second pass: union [{other['view']}] into [{k['view']}]")
            k['mask'] = k['mask'] | other['mask']
            k['views'].extend(other['views'])
            k['view'] = '+'.join(k['views'])
            _refresh(k)
            del kept[j]
            changed = True
            break
    return kept


def close_mask(mask, radius, fill_holes=False):
    """Close view-seam slivers in an equirect mask, wrapping across longitude.

    Adjacent views each stop at their own frustum edge, and back-projecting
    through a decimated grid with nearest-neighbour sampling shaves a pixel or
    two more, so unioned fragments meet along a hairline gap instead of joining.
    A morphological close welds them. The equirect is cyclic in longitude, so the
    mask is padded by wrapping columns from the far side - otherwise an object
    straddling the 180 deg seam gets a hard edge cut into it.

    fill_holes additionally floods the mask's interior, which closes the gap a
    view leaves where it saw background *through* an object (between legs, under
    an arm). It is off by default because that background is genuinely visible.
    """
    if radius < 1:
        return mask
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    src = mask.astype(np.uint8)
    pad = min(k, src.shape[1] // 2)
    wrapped = np.hstack([src[:, -pad:], src, src[:, :pad]])
    closed = cv2.morphologyEx(wrapped, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        # Flood the background from outside the frame: whatever the flood cannot
        # reach is an enclosed hole, so OR the un-flooded background back in.
        #
        # The flood has to start on a pixel that is background, and no pixel of
        # the image is guaranteed to be - a mask spanning the zenith or nadir
        # band covers a whole row, corners included. Seeding on such a pixel
        # floods nothing, every background pixel then reads as an enclosed hole,
        # and the mask swallows the entire sphere. Ringing the frame with one
        # row/column of background gives the flood a seed that is background by
        # construction, whatever the mask looks like.
        framed = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        h, w = framed.shape
        cv2.floodFill(framed, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
        holes = (framed[1:-1, 1:-1] == 0).astype(np.uint8)
        closed = closed | holes
    return closed[:, pad:pad + src.shape[1]].astype(bool)


def bbox_from_mask(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return [int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]


def draw_overlay(eq_bgr, detections, scale=1, black=False):
    """Tint + label every detection in a single blend over the full-res frame.

    Blending once instead of once per detection matters: each addWeighted over a
    5760x2880 frame is ~17M pixels of work, so N detections used to cost N full
    passes over the image.

    With black=True every detection is painted solid black and no boxes or
    labels are drawn, so the output shows only what was left unsegmented.
    """
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
               (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 165, 255)]
    if not detections:
        return eq_bgr.copy()

    h, w = eq_bgr.shape[:2]
    if black:
        merged = np.zeros((h // scale, w // scale), dtype=bool) if scale > 1 \
            else np.zeros(eq_bgr.shape[:2], dtype=bool)
        for det in detections:
            merged |= det['mask']
        if scale > 1:
            merged = cv2.resize(merged.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        out = eq_bgr.copy()
        out[merged] = 0
        return out

    tint = np.zeros((h // scale, w // scale, 3), dtype=np.uint8) if scale > 1 else np.zeros_like(eq_bgr)
    for idx, det in enumerate(detections):
        tint[det['mask']] = palette[idx % len(palette)]
    if scale > 1:
        tint = cv2.resize(tint, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(eq_bgr, 1.0, tint, 0.5, 0)

    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        box = det.get('bbox_full') or det.get('bbox')
        if box is None:
            continue
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 3)
        label = f"object {idx + 1} ({det['phrase']}) score={det['score']:.2f} [{det['view']}]"
        cv2.putText(overlay, label, (box[0], max(box[1] - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return overlay


def apply_concept_thresholds(detections, thresholds):
    """Drop detections whose phrase has its own floor and scores below it."""
    if not thresholds:
        return detections
    return [d for d in detections
            if d['score'] >= thresholds.get(d['phrase'].strip().lower(), 0.0)]


def clip_concept_pitch(detections, ranges):
    """Trim a concept's equirect mask to the band of latitudes it can occupy.

    --concept_views decides which views a concept is *asked* of, which is a
    coarse instrument: a 140 deg view centred 20 deg below the horizon still sees
    well above it. It cannot stop a mask from spreading past where the concept
    can physically be, and at these exposures 'sky' and a sunlit white wall are
    the same pixels, so one high-scoring sky mask runs straight down the wall.
    No score floor separates that - it is not a second, weaker detection, it is
    the same one. Clipping the projected mask by latitude does separate it, and
    exactly: sky is not below the horizon, whatever it looks like.

    Row 0 of the equirect grid is the zenith and the last row is the nadir, so
    pitch runs 90 -> -90 down the rows.
    """
    if not ranges:
        return detections
    out = []
    for det in detections:
        band = ranges.get(det['phrase'].strip().lower())
        if band is not None:
            lo, hi = band
            h = det['mask'].shape[0]
            pitch = 90.0 - 180.0 * (np.arange(h) + 0.5) / h
            keep = (pitch >= lo) & (pitch <= hi)
            det['mask'] = det['mask'] & keep[:, None]
            if not det['mask'].any():
                continue
            det['bbox'] = bbox_from_mask(det['mask'])
        out.append(det)
    return out


def subtract_excluded(detections, excluded):
    """Carve the excluded concepts' masks out of everything that is kept.

    Some concepts are only in the prompt to compete. Overexposed 360 frames make
    a white interior ceiling and bright sky the same pixels, so 'sky' alone
    grounds across both and no score floor separates them - the ceiling is not a
    low-confidence sky, it is a high-confidence one. Naming 'ceiling' in the
    prompt gives the grounding decoder somewhere else to put those pixels; this
    then removes them, both as detections of their own and as territory a kept
    mask claimed.

    Returns the kept detections, minus any left empty by the subtraction.
    """
    if not excluded:
        return detections
    veto = [d['mask'] for d in detections
            if d['phrase'].strip().lower() in excluded]
    kept = [d for d in detections
            if d['phrase'].strip().lower() not in excluded]
    if not veto:
        return kept
    block = np.logical_or.reduce(veto)
    out = []
    for det in kept:
        det['mask'] = det['mask'] & ~block
        if not det['mask'].any():
            continue
        det['bbox'] = bbox_from_mask(det['mask'])
        out.append(det)
    return out


def parse_concept_thresholds(spec):
    """Parse "sky=0.8,helmet=0.25" into {'sky': 0.8, 'helmet': 0.25}.

    A single box_threshold has to serve every concept in the prompt, but the
    concepts do not behave alike. Indoors, SAM3 reads a bright ceiling as 'sky'
    and scores it 0.4-0.8, overlapping nothing else, while genuine outdoor sky
    sits at 0.92+; raising the global threshold to cut the ceiling would also cut
    people. A per-concept floor separates them without touching the rest.
    """
    out = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"bad --concept_threshold entry {part!r}, expected name=value")
        name, value = part.split('=', 1)
        out[name.strip().lower()] = float(value)
    return out


def parse_concept_views(spec):
    """Parse 'sky=0:90,ground=-90:-20' into {concept: (pitch_min, pitch_max)}.

    A concept named here is only queried in views whose pitch falls in the range.
    The grounding decoder is billed once per (view, concept) pair - 6.7ms each on
    a T4 at 336px - so confining a concept that can only exist in part of the
    sphere is a straight saving, and it also stops that concept firing on
    geometry it has no business seeing (a bright floor read as sky).
    """
    ranges = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, rng = item.partition('=')
        lo, _, hi = rng.partition(':')
        try:
            ranges[name.strip()] = (float(lo), float(hi))
        except ValueError:
            raise ValueError(f"--concept_views entry {item!r} must look like 'sky=0:90'")
    return ranges


def concept_view_mask(view_list, concepts, ranges):
    """One set of allowed concept indices per view, or None if nothing is bound."""
    if not ranges:
        return None
    allowed = []
    for _, pitch in view_list:
        keep = set()
        for idx, concept in enumerate(concepts):
            lo_hi = ranges.get(concept)
            if lo_hi is None or lo_hi[0] <= pitch <= lo_hi[1]:
                keep.add(idx)
        allowed.append(keep)
    return allowed


def parse_views(spec):
    if not spec:
        return DEFAULT_VIEWS
    views = []
    for pair in spec.split(','):
        yaw, pitch = pair.strip().split(':')
        views.append((float(yaw), float(pitch)))
    return views


def coverage_report(view_list, fov_deg, grid=(180, 360)):
    """How much of the sphere the view set sees, and how deeply it overlaps.

    An object only survives back-projection whole if the views that see it
    overlap across its silhouette: a view clips whatever crosses its border, and
    the missing piece is only recovered if a neighbouring view also detects it.
    So a set can cover 100% of the sphere and still hand back ragged, partial
    masks wherever coverage is exactly one view deep - which is what a ring of
    views at a single pitch does around the nadir.

    Returns (covered_fraction, single_view_fraction), both solid-angle weighted.
    """
    eq_h, eq_w = grid
    depth = np.zeros((eq_h, eq_w), dtype=np.int16)
    for yaw, pitch in view_list:
        _, _, valid = build_inverse_maps(eq_h, eq_w, yaw, pitch, fov_deg, 256)
        depth += valid.astype(np.int16)
    w = np.broadcast_to(latitude_weights(eq_h, eq_w), (eq_h, eq_w))
    total = w.sum()
    return float(w[depth > 0].sum() / total), float(w[depth == 1].sum() / total)


def split_concepts(text_prompt):
    """Split a prompt into concepts exactly the way SAM3Detector does."""
    text_prompt_clean = text_prompt.strip()
    if ' . ' in text_prompt_clean or '.' in text_prompt_clean:
        concepts = [c.strip() for c in text_prompt_clean.replace(' . ', '.').split('.') if c.strip()]
    elif ' and ' in text_prompt_clean.lower():
        concepts = [c.strip() for c in text_prompt_clean.lower().split(' and ') if c.strip()]
    else:
        concepts = [text_prompt_clean]
    concepts = [c for c in concepts if c]
    return concepts or [text_prompt_clean]


def retune_backbone_resolution(model, resolution):
    """Rebuild the ViT's RoPE tables for a non-native input resolution.

    ``--model_resolution`` alone does not work on this SAM3 build: each global
    attention block registers a ``freqs_cis`` buffer sized from the img_size the
    model was constructed with, so feeding it anything but 1008 trips an assert
    deep in ``apply_rotary_enc``. Everything else in the ViT is already
    resolution-agnostic - the absolute pos-embed is interpolated per forward
    pass, and windowed blocks index RoPE by window rather than by image - so
    recomputing just those buffers for the new token grid is enough.

    ``rope_pt_size`` is left alone, so with rope_interp the frequencies are
    rescaled to the pretraining grid rather than extrapolated off the end of it.

    Returns True if anything was retuned.
    """
    patch = None
    for module in model.modules():
        conv = getattr(module, 'proj', None)
        if module.__class__.__name__ == 'PatchEmbed' and conv is not None:
            patch = conv.stride[0]
            break
    if not patch:
        return False
    if resolution % patch:
        raise ValueError(f"--model_resolution must be a multiple of the patch size ({patch})")

    grid = resolution // patch
    retuned = 0
    for module in model.modules():
        blocks = getattr(module, 'blocks', None)
        if blocks is None or getattr(module, 'full_attn_ids', None) is None:
            continue
        for blk in blocks:
            attn = getattr(blk, 'attn', None)
            # Windowed blocks size RoPE by their window, not by the image.
            if attn is None or getattr(blk, 'window_size', 0) != 0:
                continue
            if not getattr(attn, 'use_rope', False):
                continue
            attn.input_size = (grid, grid)
            attn._setup_rope_freqs()
            attn.freqs_cis = attn.freqs_cis.to(model.device)
            retuned += 1

    # The windowed blocks - 28 of the ViT's 32 - pad the token grid up to a whole
    # number of windows, so cost is a step function of the grid, not a smooth
    # curve. On this build (patch 14, window 24) a 336px input is one exact
    # window and runs at 38ms/view; 364px spills into a second window row and
    # jumps to 81ms, and 560px only reaches 131ms. Anything that is not a
    # multiple of window*patch is paying for tokens it then throws away.
    window = 0
    for module in model.modules():
        w = getattr(module, 'window_size', 0)
        if w:
            window = w
            break
    if window and grid % window:
        aligned = (grid // window) * window * patch or window * patch
        nxt = ((grid // window) + 1) * window * patch
        print(f"NOTE: {resolution}px gives a {grid}x{grid} token grid, which the ViT pads "
              f"out to {-(-grid // window) * window} for its windowed blocks. "
              f"{aligned}px or {nxt}px are window-aligned and much cheaper per view.")
    return retuned > 0


MASK_DECODE_BATCH = 64


def even_chunks(n, limit):
    """Split range(n) into the fewest spans of at most `limit`, sized evenly.

    Returns (start, stop) pairs. 13 views at limit 6 gives (0,7),(7,13) rather
    than the (0,6),(6,12),(12,13) a fixed stride would produce.
    """
    if n <= 0:
        return []
    limit = max(1, int(limit))
    n_chunks = -(-n // limit)                 # ceil
    base, extra = divmod(n, n_chunks)
    spans, start = [], 0
    for i in range(n_chunks):
        stop = start + base + (1 if i < extra else 0)
        spans.append((start, stop))
        start = stop
    return spans


class BatchedViewDetector:
    """Run every (view, concept) pair of a frame through SAM3 in one pass.

    SAM3's grounding stage takes a FindStage whose ``img_ids``/``text_ids`` pick,
    per query row, which encoded image and which encoded prompt to ground - the
    batching the video/training paths use. Driving it directly means the backbone
    sees the views as one batch and the text encoder sees each concept once,
    instead of views x concepts separate ``set_image`` + ``set_text_prompt``
    round-trips (each of which also re-encoded the same prompt).

    Falls back to the per-view SAM3Detector API if anything here does not line up
    with the installed SAM3 build.
    """

    def __init__(self, detector, mask_size, chunk=0, mask_threshold=0.5):
        self.detector = detector
        self.mask_size = int(mask_size)
        self.chunk = int(chunk)
        # SAM3's mask decoder outputs a soft, per-pixel confidence that a
        # pixel belongs to the detected object; this is the separate cutoff
        # that turns it into a hard mask. It is a different axis from
        # box_threshold/confidence_threshold, which only decide whether the
        # detection itself survives - an object can clear box_threshold and
        # still have its own mask come out incomplete if the decoder is only
        # moderately confident about pixels that are texture-poor or
        # atypically framed (a smooth, close-up, top-down surface, say).
        # Lowering this recovers that coverage at the cost of coarser/looser
        # mask edges everywhere, not just on the object that needed it.
        self.mask_threshold = float(mask_threshold)
        self.available = getattr(detector, 'processor', None) is not None
        self.reason = None
        # The text tower sees the same concepts on every frame, so its output is
        # frame-independent. Encoding it once saves ~33ms per frame (per chunk,
        # in fact) for the price of holding a handful of small tensors.
        self._text_cache = {}

    def _text_features(self, concepts):
        key = tuple(concepts)
        if key not in self._text_cache:
            self._text_cache[key] = self.detector.model.backbone.forward_text(
                concepts, device=self.detector.processor.device)
        return dict(self._text_cache[key])

    def _image_features(self, views_rgb):
        """Run the vision backbone over a batch of views in one upload.

        Sam3Processor.set_image_batch converts each view to a PIL image, copies
        it to the GPU on its own, and resizes it on its own, then stacks. That is
        13 host->device copies and 13 one-image kernels per frame for work that
        is identical across views. Stacking first turns it into a single copy and
        a single batched resize. The ops are the processor's own transform, in
        the same order, so the tensor handed to the backbone is unchanged.
        """
        import torch
        from torchvision.transforms import v2

        proc = self.detector.processor
        batch = torch.from_numpy(np.ascontiguousarray(np.stack(views_rgb)))
        batch = batch.to(proc.device, non_blocking=True).permute(0, 3, 1, 2)
        batch = v2.functional.resize(batch, [proc.resolution, proc.resolution])
        batch = batch.float().div_(255.0).sub_(0.5).div_(0.5)
        return self.detector.model.backbone.forward_image(batch)

    def _autocast(self):
        import torch
        use_cuda = self.detector.device == 'cuda' and torch.cuda.is_available()
        if use_cuda and getattr(self.detector, 'use_amp', False):
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return contextlib.nullcontext()

    def detect(self, views_rgb, text_prompt, box_threshold, text_threshold, prompt_mode,
               view_concepts=None, low_thr_view_idx=None, low_thr_concepts=None,
               low_thr_value=None):
        """views_rgb: list of HxWx3 uint8 RGB arrays.

        view_concepts, if given, is one set of allowed concept indices per view.
        A (view, concept) pair left out of it is never queried at all - the
        grounding decoder is billed per pair, so this is the one knob that makes
        a concept cheaper without making the others slower.

        low_thr_view_idx/low_thr_concepts/low_thr_value give a (view, concept)
        pair its own, lower box_threshold instead of the global one. This exists
        for objects like the operator's own nadir helmet: an extreme, context-free
        close-up that scores lower than a normal shot of the same object, and
        inconsistently frame to frame. Lowering --box_threshold globally to catch
        it also lets through unrelated low-confidence noise everywhere else (a red
        ladder rail read as "person", wall texture read as "sky") - scoping the
        lower floor to just the (view, concept) pairs that need it avoids that.

        Returns a list (one entry per view) of [(mask, score, phrase), ...] with
        masks as bool arrays of shape (mask_size, mask_size).
        """
        while self.available:
            try:
                return self._detect_batched(views_rgb, text_prompt, box_threshold, prompt_mode,
                                            view_concepts, low_thr_view_idx, low_thr_concepts,
                                            low_thr_value)
            except Exception as exc:
                if self._shrink_on_oom(exc, len(views_rgb)):
                    continue
                # A build mismatch (or an OOM even at one view at a time) is not
                # recoverable by resizing the batch.
                self.available = False
                self.reason = str(exc)
                print(f"  [batched inference unavailable, falling back per-view: {exc}]")
        return self._detect_serial(views_rgb, text_prompt, box_threshold, text_threshold, prompt_mode)

    def _shrink_on_oom(self, exc, n_views):
        """Halve the batch and retry when the GPU could not hold it.

        The best batch size depends on free VRAM, which depends on what else is
        on the card, so it is discovered rather than configured: the first frame
        pays for one or two failed attempts and every later frame runs at the
        size that fit.
        """
        import torch
        if not isinstance(exc, torch.cuda.OutOfMemoryError):
            return False
        current = self.chunk if self.chunk > 0 else n_views
        if current <= 1:
            return False
        # Step down by one backbone pass, not by half. The cost of a frame is
        # driven by how many passes it takes, so from 13 views the useful next
        # size is 7 (two passes) - halving to 6 buys no extra headroom over 7 but
        # forces a third pass. Halving repeatedly also overshoots: 13 -> 6 -> 3
        # skips 7, 5 and 4 entirely.
        passes = -(-n_views // current)
        # min(..., current - 1) is what makes this terminate: the pass-count
        # ladder can map a size onto itself (13 views at chunk 3 wants 3 again),
        # and without a guaranteed decrease the retry loop never ends.
        self.chunk = max(1, min(current - 1, -(-n_views // (passes + 1))))
        print(f"  [batch of {current} views did not fit in VRAM; retrying at {self.chunk}]")
        torch.cuda.empty_cache()
        return True

    def _detect_batched(self, views_rgb, text_prompt, box_threshold, prompt_mode,
                        view_concepts=None, low_thr_view_idx=None, low_thr_concepts=None,
                        low_thr_value=None):
        import torch
        import torch.nn.functional as F
        from sam3.model.data_misc import FindStage

        proc = self.detector.processor
        model = self.detector.model
        device = proc.device
        concepts = split_concepts(text_prompt)
        n_txt = len(concepts)
        thr = max(float(box_threshold), float(proc.confidence_threshold))
        # Per-pair override: skips the global confidence_threshold floor too, so
        # this can go lower than --confidence_threshold - the whole point is a
        # bar the rest of the frame doesn't get.
        has_low_thr = bool(low_thr_view_idx) and bool(low_thr_concepts) and low_thr_value is not None
        low_concept_ids = ({i for i, c in enumerate(concepts) if c.strip().lower() in low_thr_concepts}
                           if has_low_thr else set())

        results = [[] for _ in views_rgb]
        chunk = self.chunk if self.chunk > 0 else len(views_rgb)

        # Spread the views evenly over the fewest chunks that respect the limit.
        # A plain stride leaves a ragged tail - 13 views at chunk 5 runs 5+5+3 -
        # and the last, smallest pass underuses the GPU while still paying full
        # launch overhead. Even spans (5+4+4) also lower peak VRAM, which is what
        # lets _shrink_on_oom settle at a larger limit and so fewer passes.
        for start, stop in even_chunks(len(views_rgb), chunk):
            batch = views_rgb[start:stop]
            n_img = len(batch)

            with torch.inference_mode(), self._autocast():
                backbone_out = self._image_features(batch)
                backbone_out.update(self._text_features(concepts))

                # One query row per (view, concept) the caller asked for. Without
                # a restriction that is the full cross product, which is what it
                # always used to be.
                if view_concepts is None:
                    pairs = [(i, t) for i in range(n_img) for t in range(n_txt)]
                else:
                    pairs = [(i, t) for i in range(n_img)
                             for t in sorted(view_concepts[start + i])]
                if not pairs:
                    continue
                img_ids = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
                text_ids = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
                find_input = FindStage(
                    img_ids=img_ids,
                    text_ids=text_ids,
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                out = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    geometric_prompt=model._get_dummy_prompt(num_prompts=len(pairs)),
                    find_target=None,
                )

                # same scoring as Sam3Processor._forward_grounding, kept batched
                probs = out['pred_logits'].sigmoid()
                presence = out['presence_logit_dec'].sigmoid().unsqueeze(1)
                probs = (probs * presence).squeeze(-1)          # [n_img*n_txt, num_queries]

                if low_concept_ids:
                    # Per-row threshold: low_thr_value for exactly the (view,
                    # concept) pairs both named in low_thr_concepts AND at a view
                    # index in low_thr_view_idx, thr for every other row.
                    row_thr = [
                        low_thr_value
                        if (start + i) in low_thr_view_idx and t in low_concept_ids
                        else thr
                        for i, t in pairs
                    ]
                    row_thr = torch.tensor(row_thr, device=device, dtype=probs.dtype).unsqueeze(1)
                    keep = probs > row_thr
                else:
                    keep = probs > thr

                pred_masks = out['pred_masks']
                # One host sync for the whole batch instead of one per view.
                keep_cpu = keep.cpu().numpy()
                probs_cpu = probs.float().cpu().numpy()

                # Decide every selection first, then decode all of their masks
                # together. Interpolating and copying per query row cost one
                # GPU->host sync per (view, concept) pair - 39 round trips on a
                # 13-view, 3-concept frame, each one stalling the pipeline.
                # Bilinear upsampling is independent per sample, so batching
                # them changes nothing about the result.
                picks = []      # (slot index, concept index, score)
                rows = []       # the matching mask planes, still on the GPU
                filled = set()  # slots already claimed, for first_match
                for q in range(keep_cpu.shape[0]):
                    view_idx, concept_idx = pairs[q]
                    slot_i = start + view_idx
                    if prompt_mode == 'first_match' and slot_i in filled:
                        continue
                    sel = np.flatnonzero(keep_cpu[q])
                    if sel.size == 0:
                        continue
                    filled.add(slot_i)
                    for s_idx in sel:
                        picks.append((slot_i, concept_idx, float(probs_cpu[q][s_idx])))
                        rows.append(pred_masks[q][s_idx])

                # Sub-batched so a frame with many detections cannot blow up the
                # transient upsampled tensor.
                for lo in range(0, len(rows), MASK_DECODE_BATCH):
                    hi = min(lo + MASK_DECODE_BATCH, len(rows))
                    # Upsample only to the mask working resolution, not to the
                    # rendered view size - the masks are back-projected onto a
                    # decimated equirect grid anyway.
                    m = F.interpolate(
                        torch.stack(rows[lo:hi]).unsqueeze(1).float(),
                        (self.mask_size, self.mask_size),
                        mode='bilinear',
                        align_corners=False,
                    ).sigmoid()
                    m = (m > self.mask_threshold).squeeze(1).cpu().numpy()
                    for (slot_i, concept_idx, score), mask in zip(picks[lo:hi], m):
                        results[slot_i].append((mask, score, concepts[concept_idx]))

        return results

    def _detect_serial(self, views_rgb, text_prompt, box_threshold, text_threshold, prompt_mode):
        results = []
        for view_rgb in views_rgb:
            _, masks, _, scores, phrases, _ = self.detector.detect_and_segment_from_image(
                view_rgb, text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                prompt_mode=prompt_mode,
            )
            entry = []
            for idx in range(len(masks)):
                mask = masks[idx]
                if mask.ndim > 2:
                    mask = mask.squeeze()
                mask = mask > 0.5 if mask.dtype != bool else mask
                if mask.shape[0] != self.mask_size:
                    mask = cv2.resize(mask.astype(np.uint8), (self.mask_size, self.mask_size),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                entry.append((
                    mask,
                    float(scores[idx]) if idx < len(scores) else 0.0,
                    phrases[idx] if idx < len(phrases) else text_prompt,
                ))
            results.append(entry)
        return results


def load_frame(image_path, is_s3, processor):
    """Fetch and decode one equirect frame. Runs on a prefetch thread.

    S3 download and JPEG decode are ~0.2s a frame and hold no Python compute,
    so doing them one frame ahead hides them entirely behind SAM3's forward pass.
    """
    temp_dir = None
    try:
        if is_s3:
            temp_dir = tempfile.mkdtemp()
            local_path = os.path.join(temp_dir, os.path.basename(image_path))
            processor.download_image(image_path, local_path)
        else:
            local_path = image_path
        eq_bgr = cv2.imread(local_path)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    if eq_bgr is None:
        return None, None
    return eq_bgr, cv2.cvtColor(eq_bgr, cv2.COLOR_BGR2RGB)


def write_outputs(overlays_dir, masks_dir, basename, overlay, mask_u8):
    """Encode and write a frame's outputs. Runs on a writer thread."""
    cv2.imwrite(os.path.join(overlays_dir, f"{basename}.JPG"), overlay)
    Image.fromarray(mask_u8, mode="L").save(os.path.join(masks_dir, f"{basename}.png"))


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 on 360 equirectangular images via multi-view reprojection"
    )
    parser.add_argument("--input_folder", required=True,
                        help="Local folder or s3:// prefix containing equirectangular images")
    parser.add_argument("--text_prompt", required=True,
                        help="Text prompt, e.g. 'helmet . hard hat'")
    parser.add_argument("--prompt_mode", default="all", choices=["first_match", "all"])
    parser.add_argument("--output_dir", default="./outputs_equirect")
    parser.add_argument("--views", default=None,
                        help="Comma-separated yaw:pitch list, e.g. '0:-60,180:-60,0:0'. "
                             "Default covers the nadir at -60 plus the horizon ring.")
    parser.add_argument("--fov", type=float, default=110.0, help="Field of view per view (degrees)")
    parser.add_argument("--view_size", type=int, default=0,
                        help="Pixel size of each square view. 0 (default) picks 2x "
                             "--model_resolution, capped at 1024 - the backbone resizes every view "
                             "to --model_resolution anyway, so rendering much larger than that just "
                             "buys remap time and PCIe traffic. 2x keeps enough oversampling that "
                             "the downscale stays clean.")
    parser.add_argument("--confidence_threshold", type=float, default=0.25)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--concept_views", default="",
                        help="Confine a concept to views within a pitch range, e.g. "
                             "'sky=0:90'. The grounding decoder costs one pass per "
                             "(view, concept) pair, so a concept that can only appear in part "
                             "of the sphere should not be asked of every view. Concepts not "
                             "named here are asked of every view, as before.")
    parser.add_argument("--concept_mask_pitch", default="",
                        help="Clip a concept's final mask to a pitch band, e.g. 'sky=-10:90' "
                             "(pitch is +90 at the zenith, -90 at the nadir). Unlike "
                             "--concept_views, which only chooses which views a concept is "
                             "queried in, this bounds where the concept may end up on the "
                             "sphere. Use it when a concept bleeds along a surface that looks "
                             "identical to it but lies somewhere it cannot: overexposed sky "
                             "running down a white wall, say.")
    parser.add_argument("--exclude_concepts", default="",
                        help="Comma-separated concepts that are detected but never masked, e.g. "
                             "'ceiling'. Their masks are also subtracted from the concepts that "
                             "are kept. Use them as decoys: a concept that keeps stealing pixels "
                             "from a neighbouring one (indoors, 'sky' spreads across a bright "
                             "ceiling because at these exposures they are the same white) is best "
                             "fixed by giving the decoder the right name to ground those pixels "
                             "on, not by a score floor - the ceiling scores just as high as sky.")
    parser.add_argument("--concept_threshold", default="",
                        help="Per-concept score floors, e.g. 'sky=0.8,helmet=0.25'. Applied on "
                             "top of --box_threshold to concepts named here; others are "
                             "unaffected. Use it when one concept in the prompt misfires at a "
                             "score range the others do not occupy - indoors 'sky' latches onto "
                             "bright ceilings around 0.4-0.8 while real sky scores 0.92+.")
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--model_resolution", type=int, default=1008,
                        help="Square size each view is resized to before SAM3's backbone (default "
                             "1008 = the native size). Attention cost is ~quadratic in this, so it "
                             "is the only real speed dial - --view_size is not, since every view is "
                             "resized to this anyway. Cost is a STEP function, not a curve: the "
                             "ViT is windowed (patch 14, window 24), so it pads the token grid up "
                             "to a whole number of 336px windows and you pay for the padding. "
                             "Trunk time per view on a T4: 336 -> 38ms, 364 -> 81ms, 420 -> 93ms, "
                             "560 -> 131ms, 672 -> 172ms. 336 and 672 are the window-aligned sizes; "
                             "everything between 336 and 672 costs more than 336 for no more real "
                             "tokens than 672. Prefer 336, then 672.")
    parser.add_argument("--fast", action="store_true",
                        help="Preset for sub-1s frames: --model_resolution 336. Keeps all 9 views "
                             "and the rescue pass. On the sample set this tracked the 1008px masks "
                             "at ~0.87 IoU; validate on your own data before relying on it.")
    parser.add_argument("--mask_scale", type=int, default=4,
                        help="Decimation factor for the equirect grid that masks are back-projected "
                             "onto. Dedupe/area/overlap all run there, and only the final mask and "
                             "overlay are upsampled, so this is a near-quadratic saving on the CPU "
                             "side. 1 disables it (full-resolution masks).")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Cutoff on SAM3's per-pixel mask confidence (0-1) that turns the "
                             "soft predicted mask into a hard one. This is a DIFFERENT axis from "
                             "--box_threshold: box_threshold decides whether a detection survives "
                             "at all, mask_threshold decides how much of ITS mask survives once it "
                             "has. A detection can clear box_threshold comfortably and still come "
                             "back with an incomplete mask if the decoder is only moderately "
                             "confident about texture-poor or atypically framed pixels (a smooth, "
                             "close-up, top-down surface, say). Lowering this (e.g. 0.3) recovers "
                             "that coverage, at the cost of looser/coarser edges on every mask, not "
                             "just the one that needed it.")
    parser.add_argument("--mask_view_size", type=int, default=256,
                        help="View-space raster size for a predicted mask before back-projection. "
                             "SAM3 predicts masks at low resolution regardless, so upsampling past "
                             "the equirect grid's own detail buys nothing.")
    parser.add_argument("--view_batch", type=int, default=0,
                        help="Views per backbone batch (0 = all views of a frame at once). Lower "
                             "this only if the batch does not fit in VRAM.")
    parser.add_argument("--profile", action="store_true",
                        help="Print where each frame's time goes (render / inference / "
                             "backproject+dedupe / overlay+encode). Use this before tuning "
                             "--view_size, --mask_view_size or --mask_scale.")
    parser.add_argument("--no_batch_views", action="store_true",
                        help="Disable batched multi-view inference and run one SAM3 call per view "
                             "(the old, much slower path)")
    parser.add_argument("--clear_cache_each_image", action="store_true",
                        help="Call torch.cuda.empty_cache() between concepts/views. That call syncs "
                             "the device and dominated the old per-view runtime, so it is now off "
                             "by default. Enable only if you hit fragmentation.")
    parser.add_argument("--no_clear_cache_each_image", action="store_true",
                        help=argparse.SUPPRESS)  # accepted for backwards compatibility; now the default
    parser.add_argument("--merge_size_ratio", type=float, default=3.0,
                        help="Two overlapping detections merge only if the larger mask is at "
                             "most this many times the smaller. Stops a smeared near-nadir "
                             "'person' mask from swallowing the tight helmet inside it.")
    parser.add_argument("--no_rescue", action="store_true",
                        help="Disable the second-chance pass on frames that detect nothing")
    parser.add_argument("--rescue_box_threshold", type=float, default=0.15,
                        help="box_threshold for the second-chance pass; lower than the main "
                             "pass because a partly occluded object is a partial-object query")
    parser.add_argument("--rescue_text_threshold", type=float, default=0.2)
    parser.add_argument("--low_threshold_views", default="",
                        help="Pitch range 'lo:hi' (e.g. '-90:-70') where --low_threshold_concepts "
                             "use --low_threshold instead of --box_threshold. For an object that "
                             "is only ever seen in an extreme, context-free framing at one pitch "
                             "band - e.g. the operator's own helmet, always dead-center in the "
                             "nadir view - and so scores inconsistently frame to frame: lowering "
                             "--box_threshold globally to catch it also lets through unrelated "
                             "low-confidence noise everywhere else (a red ladder rail read as "
                             "'person', wall texture read as 'sky'). This scopes the lower floor "
                             "to just that band, leaving the rest of the frame at the safer "
                             "--box_threshold.")
    parser.add_argument("--low_threshold_concepts", default="",
                        help="Comma-separated concepts (must match --text_prompt) that get "
                             "--low_threshold within --low_threshold_views.")
    parser.add_argument("--low_threshold", type=float, default=0.1,
                        help="box_threshold used for --low_threshold_concepts within "
                             "--low_threshold_views.")
    parser.add_argument("--mask_close", type=int, default=2,
                        help="Radius (in mask-grid pixels) of the morphological close that welds "
                             "fragments unioned from adjacent views. 0 disables it.")
    parser.add_argument("--fill_holes", action="store_true",
                        help="Also fill enclosed holes in each merged mask (e.g. background "
                             "visible between a person's legs).")
    parser.add_argument("--dedupe_iou", type=float, default=0.4,
                        help="Containment overlap (intersection / smaller mask) above which "
                             "detections from overlapping views are merged")
    parser.add_argument("--black_overlay", action="store_true",
                        help="Paint every detected mask solid black and draw no boxes or "
                             "labels, instead of the coloured tint + annotations.")
    parser.add_argument("--save_views", action="store_true",
                        help="Also write the per-view crops and their overlays (for debugging)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--auto_fallback_to_cpu", action="store_true")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--aws_access_key_id", default=None)
    parser.add_argument("--aws_secret_access_key", default=None)
    parser.add_argument("--aws_region", default="us-east-1")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.fast and args.model_resolution == parser.get_default('model_resolution'):
        args.model_resolution = 336

    if args.view_size <= 0:
        args.view_size = min(1024, 2 * args.model_resolution)

    views = parse_views(args.views)
    mask_scale = max(1, args.mask_scale)
    concept_thresholds = parse_concept_thresholds(args.concept_threshold)
    concept_view_ranges = parse_concept_views(args.concept_views)
    prompt_concepts = split_concepts(args.text_prompt)
    mask_pitch = parse_concept_views(args.concept_mask_pitch)
    unknown_mp = set(mask_pitch) - set(prompt_concepts)
    if unknown_mp:
        raise SystemExit(
            f"--concept_mask_pitch names concepts not in --text_prompt: {sorted(unknown_mp)}")
    if mask_pitch:
        print("Per-concept mask pitch bands: " + ", ".join(
            f"{k} in [{v[0]:g},{v[1]:g}]" for k, v in mask_pitch.items()))
    excluded = {c.strip().lower() for c in args.exclude_concepts.split(',') if c.strip()}
    missing = excluded - {c.strip().lower() for c in prompt_concepts}
    if missing:
        raise SystemExit(f"--exclude_concepts names concepts not in --text_prompt: {sorted(missing)}")
    if excluded:
        print("Decoy concepts (detected, never masked): " + ", ".join(sorted(excluded)))
    unknown = set(concept_view_ranges) - set(prompt_concepts)
    if unknown:
        raise SystemExit(f"--concept_views names concepts not in --text_prompt: {sorted(unknown)}")
    if concept_view_ranges:
        print("Per-concept view ranges: " + ", ".join(
            f"{k} in pitch [{v[0]:g},{v[1]:g}]" for k, v in concept_view_ranges.items()))

    low_thr_concepts = {c.strip().lower() for c in args.low_threshold_concepts.split(',') if c.strip()}
    unknown_lt = low_thr_concepts - {c.strip().lower() for c in prompt_concepts}
    if unknown_lt:
        raise SystemExit(
            f"--low_threshold_concepts names concepts not in --text_prompt: {sorted(unknown_lt)}")
    if bool(args.low_threshold_views) != bool(low_thr_concepts):
        raise SystemExit("--low_threshold_views and --low_threshold_concepts must be given together")
    low_thr_view_idx = set()
    if args.low_threshold_views:
        lo_s, _, hi_s = args.low_threshold_views.partition(':')
        try:
            lt_lo, lt_hi = float(lo_s), float(hi_s)
        except ValueError:
            raise SystemExit("--low_threshold_views must look like 'lo:hi', e.g. '-90:-70'")
        low_thr_view_idx = {i for i, (_, pitch) in enumerate(views) if lt_lo <= pitch <= lt_hi}
        if not low_thr_view_idx:
            print(f"WARNING: --low_threshold_views [{lt_lo:g},{lt_hi:g}] matches no view in "
                  "--views; the override will never apply.")
        else:
            print(f"Low-threshold override: {sorted(low_thr_concepts)} use "
                  f"box_threshold={args.low_threshold:g} in views {sorted(low_thr_view_idx)} "
                  f"(pitch [{lt_lo:g},{lt_hi:g}])")
    covered, thin = coverage_report(views, args.fov)
    if covered < 0.999 or thin > 0.05:
        print(f"View coverage: {covered * 100:.1f}% of the sphere seen, "
              f"{thin * 100:.1f}% seen by only one view.")
        if covered < 0.999:
            print(f"  WARNING: {(1 - covered) * 100:.1f}% of the sphere is in no view at all; "
                  "objects there cannot be detected.")
        if thin > 0.05:
            print("  WARNING: large single-view regions - an object straddling a view border "
                  "there is clipped and comes back as a partial mask. Add overlapping views "
                  "(e.g. a nadir view at pitch -90) or raise --fov.")

    if concept_thresholds:
        print('Per-concept score floors: ' + ', '.join(
            f'{k}>={v}' for k, v in sorted(concept_thresholds.items())))

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # Views are a fixed shape frame to frame, so cudnn can pick kernels once.
        torch.backends.cudnn.benchmark = True

    overlays_dir = os.path.join(args.output_dir, "overlays")
    masks_dir = os.path.join(args.output_dir, "binary_masks")
    views_dir = os.path.join(args.output_dir, "views")
    for d in (overlays_dir, masks_dir):
        os.makedirs(d, exist_ok=True)
    if args.save_views:
        os.makedirs(views_dir, exist_ok=True)

    is_s3 = args.input_folder.startswith('s3://')
    processor = None
    if is_s3:
        processor = S3ImageProcessor(
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            aws_region=args.aws_region,
        )
        rest = args.input_folder[5:]
        bucket, _, prefix = rest.partition('/')
        print(f"Listing images in {args.input_folder}...")
        image_paths = list_s3_images(processor.s3_client, bucket, prefix)
    else:
        image_paths = list_local_images(args.input_folder)

    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images; {len(views)} views each")

    detector = SAM3Detector(
        hf_token=args.hf_token,
        device=device,
        confidence_threshold=args.confidence_threshold,
        auto_fallback_to_cpu=args.auto_fallback_to_cpu,
        max_image_side=0,          # views are already small; never downscale them
        clear_cache_each_image=args.clear_cache_each_image,
        model_resolution=args.model_resolution,
    )

    if args.model_resolution != 1008:
        if retune_backbone_resolution(detector.model, args.model_resolution):
            print(f"Retuned backbone RoPE tables for {args.model_resolution}px input")
        else:
            print("WARNING: could not retune RoPE tables; --model_resolution may fail")

    batched = BatchedViewDetector(detector, mask_size=args.mask_view_size, chunk=args.view_batch,
                                  mask_threshold=args.mask_threshold)
    if args.no_batch_views:
        batched.available = False

    import time
    from concurrent.futures import ThreadPoolExecutor

    summary = {}
    stage_totals = {'render': 0.0, 'infer': 0.0, 'project': 0.0, 'post': 0.0}
    map_cache = {}
    weights_cache = {}
    timings = []
    t_run_start = time.time()

    # One thread fetching the next frame, one writing the last frame's outputs.
    # Both are I/O, so they overlap with the GPU rather than competing with it.
    loader = ThreadPoolExecutor(max_workers=1)
    writer = ThreadPoolExecutor(max_workers=1)
    renderer = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))
    pending_write = None
    next_frame = loader.submit(load_frame, image_paths[0], is_s3, processor) if image_paths else None

    for n, image_path in enumerate(image_paths, 1):
        t_start = time.time()
        try:
            eq_bgr, eq_rgb = next_frame.result()
            if n < len(image_paths):
                next_frame = loader.submit(load_frame, image_paths[n], is_s3, processor)

            if eq_bgr is None:
                print(f"[{n}/{len(image_paths)}] {os.path.basename(image_path)}: unreadable, skipped")
                continue

            eq_h, eq_w = eq_bgr.shape[:2]
            # Grid the masks actually live on.
            m_h, m_w = max(1, eq_h // mask_scale), max(1, eq_w // mask_scale)
            basename = os.path.splitext(os.path.basename(image_path))[0]

            if (m_h, m_w) not in weights_cache:
                weights_cache[(m_h, m_w)] = latitude_weights(m_h, m_w)
            weights = weights_cache[(m_h, m_w)]

            def maps_for(yaw, pitch):
                key = (eq_h, eq_w, m_h, m_w, yaw, pitch, args.fov, args.view_size, args.mask_view_size)
                if key not in map_cache:
                    map_cache[key] = (
                        build_view_maps(eq_h, eq_w, yaw, pitch, args.fov, args.view_size),
                        build_inverse_maps(m_h, m_w, yaw, pitch, args.fov, args.mask_view_size),
                    )
                return map_cache[key]

            def run_views(view_list, box_thr, text_thr):
                t0 = time.time()
                view_maps = []
                jobs = []
                for yaw, pitch in view_list:
                    (map_x, map_y), inv = maps_for(yaw, pitch)
                    view_maps.append((f"y{int(yaw)}p{int(pitch)}", inv))
                    jobs.append((map_x, map_y))
                # cv2.remap drops the GIL, so the views really do render in
                # parallel. One remap off a 4096x2048 frame is ~6ms; thirteen of
                # them serially sat right in front of the GPU doing nothing.
                rendered = list(renderer.map(lambda mm: render_view(eq_rgb, mm[0], mm[1]), jobs))
                if args.save_views:
                    for (view_tag, _), view_rgb in zip(view_maps, rendered):
                        cv2.imwrite(os.path.join(views_dir, f"{basename}_{view_tag}.png"),
                                    cv2.cvtColor(view_rgb, cv2.COLOR_RGB2BGR))

                t1 = time.time()
                # low_thr_view_idx is indexed against `views` (the main pass),
                # so it only ever applies there - passing it against
                # RESCUE_VIEWS would score the wrong pitches against the
                # override, since the two lists don't share indices.
                is_main_pass = view_list is views
                per_view = batched.detect(
                    rendered, args.text_prompt, box_thr, text_thr, args.prompt_mode,
                    view_concepts=concept_view_mask(view_list, prompt_concepts,
                                                    concept_view_ranges),
                    low_thr_view_idx=low_thr_view_idx if is_main_pass else None,
                    low_thr_concepts=low_thr_concepts if is_main_pass else None,
                    low_thr_value=args.low_threshold if is_main_pass else None)
                t2 = time.time()

                found = []
                for (view_tag, (inv_x, inv_y, valid)), dets in zip(view_maps, per_view):
                    for mask, score, phrase in dets:
                        eq_mask = mask_to_equirect(mask, inv_x, inv_y, valid)
                        if eq_mask is None:
                            continue
                        # An object mask touching its source view's own border
                        # is cut off by that view's frustum, not by the object's
                        # real extent - the rest of it lives in whatever view is
                        # next door. Flag it so we know to go get that part too.
                        clipped = bool(mask[0, :].any() or mask[-1, :].any()
                                       or mask[:, 0].any() or mask[:, -1].any())
                        found.append({
                            'mask': eq_mask,
                            'score': score,
                            'phrase': phrase,
                            'view': view_tag,
                            'clipped': clipped,
                        })
                stage_totals['render'] += t1 - t0
                stage_totals['infer'] += t2 - t1
                stage_totals['project'] += time.time() - t2
                return found

            detections = run_views(views, args.box_threshold, args.text_threshold)
            detections = apply_concept_thresholds(detections, concept_thresholds)

            # Only frames that came back empty - or whose only hits are cut off
            # by their own view's edge - pay for the extra, seam-offset views.
            # On a set where most frames already detect cleanly, this still
            # costs almost nothing.
            rescued = False
            needs_rescue = not detections or any(d.get('clipped') for d in detections)
            if needs_rescue and not args.no_rescue:
                rescue_detections = run_views(RESCUE_VIEWS, args.rescue_box_threshold,
                                              args.rescue_text_threshold)
                rescue_detections = apply_concept_thresholds(rescue_detections, concept_thresholds)
                rescued = bool(rescue_detections)
                # Add to, rather than replace, an already-partial detection -
                # dedupe() below merges the fragments back into one object.
                detections = detections + rescue_detections

            t_post = time.time()
            detections = dedupe(detections, args.dedupe_iou, args.merge_size_ratio, weights)

            # Before the close, so it cannot bridge back across the cut.
            detections = clip_concept_pitch(detections, mask_pitch)

            # Weld the seams left where fragments from adjacent views meet. Done
            # after dedupe so the close operates on the full unioned object.
            if args.mask_close > 0 or args.fill_holes:
                for det in detections:
                    det['mask'] = close_mask(det['mask'], args.mask_close, args.fill_holes)
                    det['bbox'] = bbox_from_mask(det['mask'])

            detections = subtract_excluded(detections, excluded)

            for det in detections:
                box = det['bbox']
                det['bbox_full'] = ([int(round(c * mask_scale)) for c in box]
                                    if box is not None else None)

            # Compositing the overlay and upscaling the union mask is pure CPU
            # work on a full-resolution frame, so it rides along with the encode
            # on the writer thread instead of stalling the next frame's GPU pass.
            def finish(eq_bgr=eq_bgr, detections=detections, basename=basename,
                       m_h=m_h, m_w=m_w, eq_h=eq_h, eq_w=eq_w):
                overlay = draw_overlay(eq_bgr, detections, scale=mask_scale,
                                       black=args.black_overlay)
                merged = np.zeros((m_h, m_w), dtype=bool)
                for det in detections:
                    merged |= det['mask']
                merged_u8 = merged.astype(np.uint8) * 255
                if mask_scale > 1:
                    merged_u8 = cv2.resize(merged_u8, (eq_w, eq_h),
                                           interpolation=cv2.INTER_NEAREST)
                write_outputs(overlays_dir, masks_dir, basename, overlay, merged_u8)

            # Let the previous frame's encode finish, then hand this one off.
            if pending_write is not None:
                pending_write.result()
            pending_write = writer.submit(finish)

            stage_totals['post'] += time.time() - t_post

            summary[os.path.basename(image_path)] = [
                {'score': d['score'], 'phrase': d['phrase'], 'view': d['view'],
                 'bbox': d['bbox_full'], 'rescued': rescued}
                for d in detections
            ]
            elapsed = time.time() - t_start
            timings.append(elapsed)
            tag = f" (rescued)" if rescued else ""
            tag += f" [{elapsed:.2f}s]"
            print(f"[{n}/{len(image_paths)}] {os.path.basename(image_path)}: {len(detections)} objects{tag}")

        except Exception as exc:
            print(f"[{n}/{len(image_paths)}] {os.path.basename(image_path)}: FAILED - {exc}")

    if pending_write is not None:
        pending_write.result()
    loader.shutdown()
    writer.shutdown()
    renderer.shutdown()

    with open(os.path.join(args.output_dir, "predictions_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    total = sum(len(v) for v in summary.values())
    hit_frames = sum(1 for v in summary.values() if v)
    rescued_frames = sum(1 for v in summary.values() if v and v[0].get('rescued'))
    print(f"\nDone. {total} detections across {hit_frames}/{len(summary)} frames "
          f"({rescued_frames} recovered by the second-chance pass).")
    if args.profile and timings:
        n_frames = len(timings)
        print("Stage means per frame: " + ", ".join(
            f"{k} {v / n_frames:.3f}s" for k, v in stage_totals.items()))
    if timings:
        print(f"Per-image: mean {sum(timings)/len(timings):.2f}s, "
              f"min {min(timings):.2f}s, max {max(timings):.2f}s")
    total_elapsed = time.time() - t_run_start
    h, rem = divmod(total_elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Total run time: {int(h)}h {int(m)}m {s:.1f}s ({total_elapsed:.1f}s) "
          f"for {len(summary)} images")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
