# -*- coding: utf-8 -*-
'''
Author: Naiyuan liu (patched for InsightFace 0.7.x)
'''
from __future__ import division
import collections, glob, os
import os.path as osp
import numpy as np
import cv2

try:
    from insightface.model_zoo import get_model as _get_model
except Exception:
    _get_model=None
from insightface.app import FaceAnalysis
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding', 'landmark'
])
Face.__new__.__defaults__ = (None,) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        self.mode = 'None'

        # Try local ONNX first: <root>/<name>/*.onnx
        root_expanded = os.path.expanduser(root) if root else None
        onnx_dir = osp.join(root_expanded, name) if root_expanded else None
        onnx_files = sorted(glob.glob(osp.join(onnx_dir, '*.onnx'))) if (onnx_dir and os.path.isdir(onnx_dir)) else []

        if onnx_files:
            for onnx_file in onnx_files:
                if '_selfgen_' in onnx_file:
                    continue
                model = _get_model(onnx_file)
                if model.taskname not in self.models:
                    print('find model:', onnx_file, model.taskname)
                    self.models[model.taskname] = model
                else:
                    print('duplicated model task type, ignore:', onnx_file, model.taskname)
                    del model
            assert 'detection' in self.models, f"No detection model under {onnx_dir}"
            self.det_model = self.models['detection']
            self._backend = 'onnx'
        else:
            # Fallback to a maintained model pack (map legacy names)
            legacy_map = {'antelope': 'buffalo_l', 'antelopev2': 'buffalo_l'}
            pack = legacy_map.get(name, name or 'buffalo_l')
            cache_root = os.path.expanduser('~/.insightface')  # must be a string
            print(f"No local ONNX found; using FaceAnalysis pack '{pack}' with cache '{cache_root}'")
            self.app = FaceAnalysis(
                name=pack,
                root=cache_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # add this
            )
            # Pre-prepare once; caller may override det_size later
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.models = self.app.models
            assert 'detection' in self.models, "Pack missing detection model"
            self.det_model = self.models['detection']
            self._backend = 'pack'

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                # InsightFace 0.7.x detectors take threshold in prepare, not detect()
                try:
                    model.prepare(ctx_id, det_thresh=det_thresh, input_size=det_size)
                except TypeError:
                    model.prepare(ctx_id, input_size=det_size)
                    if hasattr(model, 'threshold'):
                        model.threshold = det_thresh
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        # Call detect() without 'threshold' kwarg (0.7.x API)
        try:
            bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        except TypeError:
            try:
                bboxes, kpss = self.det_model.detect(img, max_num=max_num)
            except TypeError:
                bboxes, kpss = self.det_model.detect(img)

        if bboxes is None or (hasattr(bboxes, 'shape') and bboxes.shape[0] == 0):
            return None

        det_score = bboxes[..., 4] if hasattr(bboxes, 'ndim') and bboxes.ndim == 2 and bboxes.shape[1] >= 5 else None
        best_index = int(np.argmax(det_score)) if det_score is not None else 0

        kps = kpss[best_index] if kpss is not None else None
        M, _ = face_align.estimate_norm(kps, crop_size, mode=self.mode)
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        return [align_img], [M]
