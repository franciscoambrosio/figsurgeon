"""Segment the eight boxes with a SAM-class model and save the masks as PNGs.

Needs torch and transformers, which the package does not depend on, so it runs in a
throwaway environment:

    uv venv --python 3.11 /tmp/samenv
    uv pip install --python /tmp/samenv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install --python /tmp/samenv/bin/python transformers pillow numpy scikit-image
    /tmp/samenv/bin/python evals/segmentation_models/run_models.py facebook/sam2.1-hiera-tiny sam2tiny

Models measured (all Apache-2.0, all box-prompted, all CPU):
    facebook/sam2.1-hiera-tiny   sam2tiny    ~150 MB   the pick
    nielsr/slimsam-77-uniform    slimsam     ~40 MB
    facebook/sam-vit-base        samvitb     ~375 MB   3x slower and not better
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Sam2Model, SamModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C                                                        # noqa: E402

_LOADED = {}


def load(model_id):
    if model_id not in _LOADED:
        cls = Sam2Model if 'sam2' in model_id else SamModel
        _LOADED[model_id] = (cls.from_pretrained(model_id).eval(),
                             AutoProcessor.from_pretrained(model_id))
    return _LOADED[model_id]


def mask_from_box(img, box, model_id):
    """SAM takes the box directly; the processor handles the coordinate transform."""
    model, proc = load(model_id)
    img = img.convert('RGB')
    inputs = proc(images=img, input_boxes=[[[float(v) for v in box]]], return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    sizes = inputs.get('original_sizes')
    if sizes is None:
        sizes = torch.tensor([[img.size[1], img.size[0]]])
    try:
        masks = proc.post_process_masks(out.pred_masks.cpu(), sizes)[0]
    except TypeError:                       # SAM 1's processor takes the reshaped size too
        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), sizes, inputs['reshaped_input_sizes'].cpu())[0]
    masks = masks[0] if masks.ndim == 4 else masks
    scores = np.ravel(out.iou_scores.cpu().numpy())
    best = int(np.argmax(scores))
    return np.asarray(masks[best]).astype(bool), float(scores[best])


def run(model_id, tag):
    out_dir = os.path.join(C.HERE, f'masks_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    load(model_id)                    # warm, so the timings below are work and not download
    timings = {}
    for case in C.load():
        img = C.image(case)
        t = time.time()
        mask, score = mask_from_box(img, case['box'], model_id)
        seconds = time.time() - t
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(out_dir, case['id'] + '.png'))
        timings[case['id']] = {'seconds': seconds, 'score': score}
        print(f"{case['id']:18} {seconds:6.2f}s  mask {mask.mean():5.1%}  score {score:.2f}")
    json.dump(timings, open(os.path.join(out_dir, 'timings.json'), 'w'), indent=1)
    print('masks in', out_dir)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
