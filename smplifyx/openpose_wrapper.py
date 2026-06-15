# -*- coding: utf-8 -*-
"""OpenPose-compatible keypoint detection using controlnet_aux's OpenposeDetector.

The original CMU OpenPose (Caffe-based) model servers are unreliable, so we use
controlnet_aux, which re-implements the same CMU OpenPose neural network in
PyTorch and downloads model weights from Hugging Face automatically.

The public API is identical to the playing_smplifyx openpose_wrapper.py:
    datum = detect_keypoints(image_bgr_np)
    datum.poseKeypoints  -> (N, 25, 3) BODY_25 body keypoints [x, y, conf]
    datum.handKeypoints  -> [left (N,21,3), right (N,21,3)]
    datum.faceKeypoints  -> (N, 70, 3)
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from controlnet_aux.open_pose import OpenposeDetector

# Download PyTorch OpenPose weights from lllyasviel/ControlNet on Hugging Face
# (body_pose_model.pth, hand_pose_model.pth, facenet.pth)
_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

# ---- COCO-18 → BODY_25 index mapping ----------------------------------------
# COCO-18:  0=Nose 1=Neck 2=RSho 3=RElb 4=RWri 5=LSho 6=LElb 7=LWri
#           8=RHip 9=RKne 10=RAn 11=LHip 12=LKne 13=LAn 14=REye 15=LEye
#           16=REar 17=LEar
# BODY_25:  0=Nose 1=Neck 2=RSho 3=RElb 4=RWri 5=LSho 6=LElb 7=LWri
#           8=MidHip 9=RHip 10=RKne 11=RAn 12=LHip 13=LKne 14=LAn 15=REye
#           16=LEye 17=REar 18=LEar 19=LBigToe 20=LSmToe 21=LHeel
#           22=RBigToe 23=RSmToe 24=RHeel

# BODY_25 indices that map directly from COCO-18
_COCO18_FROM_BODY25 = {
    0: 0,   # Nose
    1: 1,   # Neck
    2: 2,   # RShoulder
    3: 3,   # RElbow
    4: 4,   # RWrist
    5: 5,   # LShoulder
    6: 6,   # LElbow
    7: 7,   # LWrist
    9: 8,   # RHip
    10: 9,  # RKnee
    11: 10, # RAnkle
    12: 11, # LHip
    13: 12, # LKnee
    14: 13, # LAnkle
    15: 14, # REye
    16: 15, # LEye
    17: 16, # REar
    18: 17, # LEar
    # BODY_25[8] (MidHip) is computed as midpoint of COCO18[8] and COCO18[11]
    # BODY_25[19:25] (feet) left as zero — not detected
}


def _keypoints_to_array(kp_list, n_joints, H, W):
    """Convert a list of Keypoint namedtuples (or None) to an (n_joints, 3) array.

    Keypoints have normalised x/y ∈ [0,1]; we convert to pixel coordinates.
    Undetected keypoints (None) or sentinel (-1,-1) are returned with confidence 0.
    """
    arr = np.zeros((n_joints, 3), dtype=np.float32)
    for i, kp in enumerate(kp_list):
        if i >= n_joints:
            break
        if kp is not None and kp.x >= 0 and kp.y >= 0:
            arr[i, 0] = kp.x * W
            arr[i, 1] = kp.y * H
            arr[i, 2] = float(kp.score) if kp.score is not None else 1.0
    return arr


def _coco18_to_body25(coco18, H, W):
    """Convert a list of 18 COCO Keypoints to a (25, 3) BODY_25 array."""
    body25 = np.zeros((25, 3), dtype=np.float32)

    for b25_idx, c18_idx in _COCO18_FROM_BODY25.items():
        kp = coco18[c18_idx] if c18_idx < len(coco18) else None
        if kp is not None and kp.x >= 0 and kp.y >= 0:
            body25[b25_idx, 0] = kp.x * W
            body25[b25_idx, 1] = kp.y * H
            body25[b25_idx, 2] = float(kp.score) if kp.score is not None else 1.0

    # MidHip (index 8) = midpoint of RHip (COCO18[8]) and LHip (COCO18[11])
    r_hip = coco18[8] if len(coco18) > 8 else None
    l_hip = coco18[11] if len(coco18) > 11 else None
    if (r_hip is not None and l_hip is not None and
            r_hip.x >= 0 and r_hip.y >= 0 and l_hip.x >= 0 and l_hip.y >= 0):
        body25[8, 0] = (r_hip.x + l_hip.x) * 0.5 * W
        body25[8, 1] = (r_hip.y + l_hip.y) * 0.5 * H
        body25[8, 2] = min(
            float(r_hip.score) if r_hip.score is not None else 1.0,
            float(l_hip.score) if l_hip.score is not None else 1.0,
        )

    return body25


def _remap_hf_to_timm(hf_weights):
    """Convert rizvandwiki/gender-classification Transformers ViT weights to timm layout.

    The key difference is that HF stores Q, K, V as separate linear layers while
    timm merges them into a single QKV projection (shape [3*D, D]).
    """
    timm_sd = {}

    simple = {
        'classifier.weight':                              'head.weight',
        'classifier.bias':                                'head.bias',
        'vit.embeddings.cls_token':                       'cls_token',
        'vit.embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
        'vit.embeddings.patch_embeddings.projection.bias':   'patch_embed.proj.bias',
        'vit.embeddings.position_embeddings':             'pos_embed',
        'vit.layernorm.weight':                           'norm.weight',
        'vit.layernorm.bias':                             'norm.bias',
    }
    for hf_key, timm_key in simple.items():
        timm_sd[timm_key] = hf_weights[hf_key]

    for i in range(12):
        hf = f'vit.encoder.layer.{i}'
        tm = f'blocks.{i}'

        timm_sd[f'{tm}.norm1.weight'] = hf_weights[f'{hf}.layernorm_before.weight']
        timm_sd[f'{tm}.norm1.bias']   = hf_weights[f'{hf}.layernorm_before.bias']

        q_w = hf_weights[f'{hf}.attention.attention.query.weight']
        k_w = hf_weights[f'{hf}.attention.attention.key.weight']
        v_w = hf_weights[f'{hf}.attention.attention.value.weight']
        timm_sd[f'{tm}.attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_weights[f'{hf}.attention.attention.query.bias']
        k_b = hf_weights[f'{hf}.attention.attention.key.bias']
        v_b = hf_weights[f'{hf}.attention.attention.value.bias']
        timm_sd[f'{tm}.attn.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        timm_sd[f'{tm}.attn.proj.weight'] = hf_weights[f'{hf}.attention.output.dense.weight']
        timm_sd[f'{tm}.attn.proj.bias']   = hf_weights[f'{hf}.attention.output.dense.bias']

        timm_sd[f'{tm}.norm2.weight'] = hf_weights[f'{hf}.layernorm_after.weight']
        timm_sd[f'{tm}.norm2.bias']   = hf_weights[f'{hf}.layernorm_after.bias']

        timm_sd[f'{tm}.mlp.fc1.weight'] = hf_weights[f'{hf}.intermediate.dense.weight']
        timm_sd[f'{tm}.mlp.fc1.bias']   = hf_weights[f'{hf}.intermediate.dense.bias']

        timm_sd[f'{tm}.mlp.fc2.weight'] = hf_weights[f'{hf}.output.dense.weight']
        timm_sd[f'{tm}.mlp.fc2.bias']   = hf_weights[f'{hf}.output.dense.bias']

    return timm_sd


def _crop_person(image_rgb, kps_body25, pad_frac=0.15):
    """Return a tight bounding-box crop of one person using their BODY_25 keypoints.

    kps_body25: (25, 3) pixel-coord array (x, y, conf).
    Returns an H×W×3 uint8 RGB array, or None if fewer than 2 joints are visible.
    """
    visible = kps_body25[kps_body25[:, 2] > 0.1]
    if len(visible) < 2:
        return None
    H, W = image_rgb.shape[:2]
    x0 = visible[:, 0].min()
    y0 = visible[:, 1].min()
    x1 = visible[:, 0].max()
    y1 = visible[:, 1].max()
    pad_x = (x1 - x0) * pad_frac
    pad_y = (y1 - y0) * pad_frac
    x0 = max(0, int(x0 - pad_x))
    y0 = max(0, int(y0 - pad_y))
    x1 = min(W, int(x1 + pad_x))
    y1 = min(H, int(y1 + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    return image_rgb[y0:y1, x0:x1]


class _GenderClassifier:
    """ViT-B/16 gender classifier loaded from rizvandwiki/gender-classification.

    Weights are remapped from HuggingFace Transformers format to timm format so
    no `transformers` dependency is required — only timm + safetensors.
    """

    _REPO   = 'rizvandwiki/gender-classification'
    _LABELS = ['female', 'male']   # id2label: {0: female, 1: male}

    def __init__(self):
        import timm
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        print('Loading gender classifier from HuggingFace Hub ...')
        weights_path = hf_hub_download(self._REPO, 'model.safetensors')
        hf_weights = safetensors.torch.load_file(weights_path)

        self._model = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=2)
        self._model.load_state_dict(_remap_hf_to_timm(hf_weights))
        self._model.eval()

    def predict(self, image_rgb_uint8, kps_body25):
        """Return 'female' or 'male' for the person described by kps_body25.

        Falls back to 'neutral' when the crop is too small to be reliable.
        """
        crop = _crop_person(image_rgb_uint8, kps_body25)
        if crop is None:
            return 'neutral'

        # Pad to square before resizing to avoid aspect-ratio distortion
        h, w = crop.shape[:2]
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[(side - h) // 2:(side - h) // 2 + h,
               (side - w) // 2:(side - w) // 2 + w] = crop
        img = Image.fromarray(padded).resize((224, 224), Image.BILINEAR)
        t = TF.to_tensor(img)                            # [0, 1]
        t = TF.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # [-1, 1]
        t = t.unsqueeze(0)

        with torch.no_grad():
            pred = self._model(t).argmax(dim=1).item()
        return self._LABELS[pred]


_gender_clf = None   # loaded lazily on first call to detect_keypoints


class _Datum:
    """Minimal stand-in for pyopenpose's Datum; holds the detected arrays."""
    def __init__(self, pose_kps, hand_kps_left, hand_kps_right, face_kps,
                 gender_pd=None):
        self.poseKeypoints = pose_kps          # (N, 25, 3)
        self.handKeypoints = [hand_kps_left,   # (N, 21, 3)
                              hand_kps_right]  # (N, 21, 3)
        self.faceKeypoints = face_kps          # (N, 70, 3)
        self.gender_pd = gender_pd or []       # list[str] length N


def detect_keypoints(image_bgr):
    """Detect all-person body, hand, face keypoints and predict gender.

    Returns a _Datum whose .poseKeypoints / .handKeypoints / .faceKeypoints
    match the shape conventions expected by pipeline.py / datum_to_keypoints(),
    and .gender_pd is a list[str] of 'female'/'male' predictions per person.
    """
    global _gender_clf

    H, W = image_bgr.shape[:2]

    poses = _detector.detect_poses(
        image_bgr,
        include_hand=True,
        include_face=True,
    )

    if not poses:
        empty = np.zeros((0, 1, 3), dtype=np.float32)
        return _Datum(empty, empty, empty, empty, gender_pd=[])

    n = len(poses)
    pose_kps  = np.zeros((n, 25, 3), dtype=np.float32)
    lhand_kps = np.zeros((n, 21, 3), dtype=np.float32)
    rhand_kps = np.zeros((n, 21, 3), dtype=np.float32)
    face_kps  = np.zeros((n, 70, 3), dtype=np.float32)

    for i, pose in enumerate(poses):
        pose_kps[i] = _coco18_to_body25(pose.body.keypoints, H, W)
        if pose.left_hand is not None:
            lhand_kps[i] = _keypoints_to_array(pose.left_hand, 21, H, W)
        if pose.right_hand is not None:
            rhand_kps[i] = _keypoints_to_array(pose.right_hand, 21, H, W)
        if pose.face is not None:
            face_kps[i] = _keypoints_to_array(pose.face, 70, H, W)

    # Gender classification — load model once, then predict per person
    if _gender_clf is None:
        _gender_clf = _GenderClassifier()

    image_rgb = image_bgr[:, :, ::-1].copy()   # BGR → RGB uint8
    gender_pd = [_gender_clf.predict(image_rgb, pose_kps[i]) for i in range(n)]
    print(f'Gender predictions: {gender_pd}')

    return _Datum(pose_kps, lhand_kps, rhand_kps, face_kps, gender_pd=gender_pd)
