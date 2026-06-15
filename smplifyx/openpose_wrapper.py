# -*- coding: utf-8 -*-
"""OpenPose-compatible keypoint detection using rtmlib's Wholebody detector.

rtmlib uses onnxruntime and downloads RTMPose ONNX weights from openmmlab
automatically.  With to_openpose=True the Wholebody model outputs 134
keypoints in OpenPose-134 format:
    [0:18]   OpenPose-18 body joints
    [18:24]  6 foot joints (LBigToe, LSmToe, LHeel, RBigToe, RSmToe, RHeel)
    [24:92]  68 face joints
    [92:113] 21 left-hand joints
    [113:134] 21 right-hand joints

This maps directly to BODY_25 (only MidHip must be computed as the midpoint
of RHip and LHip).  All 25 body joints — including the 6 foot joints — are
natively detected rather than zeroed.

Public API (identical to the pyopenpose wrapper):
    datum = detect_keypoints(image_bgr_np)
    datum.poseKeypoints   -> (N, 25, 3)  BODY_25  [x, y, confidence]
    datum.handKeypoints   -> [left (N,21,3), right (N,21,3)]
    datum.faceKeypoints   -> (N, 70, 3)
    datum.gender_pd       -> list[str] length N
"""

import numpy as np
import torch
from PIL import Image
from rtmlib import Wholebody

_wholebody = Wholebody(to_openpose=True, backend='onnxruntime', device='cpu')

# OpenPose-134 indices after to_openpose=True conversion:
#   OP134 body-18:  0=Nose 1=Neck 2=RSho 3=RElb 4=RWri 5=LSho 6=LElb 7=LWri
#                   8=RHip 9=RKne 10=RAn 11=LHip 12=LKne 13=LAn
#                   14=REye 15=LEye 16=REar 17=LEar
#   OP134 feet-6:   18=LBigToe 19=LSmToe 20=LHeel 21=RBigToe 22=RSmToe 23=RHeel
#
# BODY_25:          0=Nose 1=Neck 2=RSho 3=RElb 4=RWri 5=LSho 6=LElb 7=LWri
#                   8=MidHip 9=RHip 10=RKne 11=RAn 12=LHip 13=LKne 14=LAn
#                   15=REye 16=LEye 17=REar 18=LEar
#                   19=LBigToe 20=LSmToe 21=LHeel 22=RBigToe 23=RSmToe 24=RHeel

# Direct per-joint mapping (b25_idx, op134_idx); MidHip (8) is computed below.
_B25_FROM_OP134 = [
    (0,  0),   # Nose
    (1,  1),   # Neck
    (2,  2),   # RShoulder
    (3,  3),   # RElbow
    (4,  4),   # RWrist
    (5,  5),   # LShoulder
    (6,  6),   # LElbow
    (7,  7),   # LWrist
    # 8 = MidHip — computed as midpoint of RHip (8) and LHip (11)
    (9,  8),   # RHip
    (10, 9),   # RKnee
    (11, 10),  # RAnkle
    (12, 11),  # LHip
    (13, 12),  # LKnee
    (14, 13),  # LAnkle
    (15, 14),  # REye
    (16, 15),  # LEye
    (17, 16),  # REar
    (18, 17),  # LEar
    (19, 18),  # LBigToe
    (20, 19),  # LSmallToe
    (21, 20),  # LHeel
    (22, 21),  # RBigToe
    (23, 22),  # RSmallToe
    (24, 23),  # RHeel
]
_B25_IDX = np.array([b for b, _ in _B25_FROM_OP134], dtype=np.intp)
_OP_IDX  = np.array([o for _, o in _B25_FROM_OP134], dtype=np.intp)


def _op134_to_body25(keypoints, scores):
    """Convert (N, 134, 2) + (N, 134) → (N, 25, 3) BODY_25 array."""
    n = keypoints.shape[0]
    body25 = np.zeros((n, 25, 3), dtype=np.float32)
    body25[:, _B25_IDX, :2] = keypoints[:, _OP_IDX]
    body25[:, _B25_IDX,  2] = scores[:, _OP_IDX]
    # MidHip = midpoint of RHip (OP134[8]) and LHip (OP134[11])
    r_sc = scores[:, 8]
    l_sc = scores[:, 11]
    valid = (r_sc > 0) & (l_sc > 0)
    body25[valid, 8, :2] = (keypoints[valid, 8] + keypoints[valid, 11]) * 0.5
    body25[valid, 8,  2] = np.minimum(r_sc[valid], l_sc[valid])
    return body25


def _extract_hands(keypoints, scores):
    """Return (N,21,3) left and (N,21,3) right hand arrays from OP134 output."""
    n = keypoints.shape[0]
    left  = np.zeros((n, 21, 3), dtype=np.float32)
    right = np.zeros((n, 21, 3), dtype=np.float32)
    left[:,  :, :2] = keypoints[:, 92:113]
    left[:,  :,  2] = scores[:,   92:113]
    right[:, :, :2] = keypoints[:, 113:134]
    right[:, :,  2] = scores[:,    113:134]
    return left, right


def _extract_face(keypoints, scores, body25):
    """Return (N,70,3) face array.

    RTMPose outputs 68 face keypoints; OpenPose expects 70.  The last two
    slots are LEye and REye, taken from the already-computed body25 array.
    """
    n = keypoints.shape[0]
    face = np.zeros((n, 70, 3), dtype=np.float32)
    face[:, :68, :2] = keypoints[:, 24:92]
    face[:, :68,  2] = scores[:,   24:92]
    # slot 68 = LEye (BODY_25[16]), slot 69 = REye (BODY_25[15])
    face[:, 68] = body25[:, 16]
    face[:, 69] = body25[:, 15]
    return face


def _crop_person(image_rgb, body25, padding=0.15):
    """Return a tight crop around the person described by body25 (25,3)."""
    visible = body25[body25[:, 2] > 0.1]
    if len(visible) < 2:
        return None
    h, w = image_rgb.shape[:2]
    x0, y0 = visible[:, 0].min(), visible[:, 1].min()
    x1, y1 = visible[:, 0].max(), visible[:, 1].max()
    px = (x1 - x0) * padding
    py = (y1 - y0) * padding
    x0 = max(0,  int(x0 - px))
    y0 = max(0,  int(y0 - py))
    x1 = min(w,  int(x1 + px))
    y1 = min(h,  int(y1 + py))
    if x1 <= x0 or y1 <= y0:
        return None
    return image_rgb[y0:y1, x0:x1]


class _GenderClassifier:
    _REPO = 'rizvandwiki/gender-classification'

    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print('Loading gender classifier from HuggingFace Hub ...')
        self._processor = AutoImageProcessor.from_pretrained(self._REPO)
        self._model = AutoModelForImageClassification.from_pretrained(self._REPO)
        self._model.eval()

    def predict(self, image_rgb, body25_kp):
        crop = _crop_person(image_rgb, body25_kp)
        if crop is None:
            return 'neutral'
        inputs = self._processor(images=Image.fromarray(crop), return_tensors='pt')
        with torch.no_grad():
            logits = self._model(**inputs).logits
        idx = logits.argmax(-1).item()
        return self._model.config.id2label[idx]


_gender_clf = None


class _Datum:
    def __init__(self, pose_kp, left_hand_kp, right_hand_kp, face_kp,
                 predicted_genders=None):
        self.poseKeypoints  = pose_kp                              # (N, 25, 3)
        self.handKeypoints  = [left_hand_kp, right_hand_kp]       # (N, 21, 3) each
        self.faceKeypoints  = face_kp                              # (N, 70, 3)
        self.gender_pd      = predicted_genders or []              # list[str]


def detect_keypoints(image_bgr):
    """Detect body, hand, and face keypoints and predict gender.

    Returns a _Datum whose arrays match the shapes expected by pipeline.py /
    datum_to_keypoints(), with all BODY_25 joints (including foot joints
    19-24) natively detected rather than zeroed.
    """
    global _gender_clf

    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        empty = np.zeros((0, 1, 3), dtype=np.float32)
        return _Datum(empty, empty, empty, empty)

    keypoints, scores = _wholebody(image_bgr)   # (N, 134, 2), (N, 134)

    if keypoints.shape[0] == 0:
        empty = np.zeros((0, 1, 3), dtype=np.float32)
        return _Datum(empty, empty, empty, empty, predicted_genders=[])

    body25              = _op134_to_body25(keypoints, scores)
    left_hand, r_hand   = _extract_hands(keypoints, scores)
    face                = _extract_face(keypoints, scores, body25)

    if _gender_clf is None:
        _gender_clf = _GenderClassifier()

    image_rgb = image_bgr[:, :, ::-1].copy()
    n = body25.shape[0]
    predicted_genders = [
        _gender_clf.predict(image_rgb, body25[i]) for i in range(n)]
    print(f'Gender predictions: {predicted_genders}')

    return _Datum(body25, left_hand, r_hand, face,
                  predicted_genders=predicted_genders)
