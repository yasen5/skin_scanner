# -*- coding: utf-8 -*-
"""OpenPose-compatible keypoint detection using controlnet_aux's OpenposeDetector.

The original CMU OpenPose (Caffe-based) model servers are unreliable, so we use
controlnet_aux, which re-implements the same CMU OpenPose neural network in
PyTorch and downloads model weights from Hugging Face automatically.

The public API is identical to the playing_smplifyx openpose_wrapper.py:
    openpose_result = detect_keypoints(image_bgr_np)
    openpose_result.poseKeypoints  -> (N, 25, 3) BODY_25 body keypoints [x, y, confidence]
    openpose_result.handKeypoints  -> [left (N,21,3), right (N,21,3)]
    openpose_result.faceKeypoints  -> (N, 70, 3)
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


def _keypoints_to_array(keypoint_list, joint_count, image_height, image_width):
    """Convert a list of Keypoint namedtuples (or None) to an (n_joints, 3) array.

    Keypoints have normalised x/y ∈ [0,1]; we convert to pixel coordinates.
    Undetected keypoints (None) or sentinel (-1,-1) are returned with confidence 0.
    """
    keypoint_array = np.zeros((joint_count, 3), dtype=np.float32)
    for joint_index, keypoint in enumerate(keypoint_list):
        if joint_index >= joint_count:
            break
        if keypoint is not None and keypoint.x >= 0 and keypoint.y >= 0:
            keypoint_array[joint_index, 0] = keypoint.x * image_width
            keypoint_array[joint_index, 1] = keypoint.y * image_height
            keypoint_array[joint_index, 2] = (
                float(keypoint.score) if keypoint.score is not None else 1.0)
    return keypoint_array


def _coco18_to_body25(coco18_keypoints, image_height, image_width):
    """Convert a list of 18 COCO Keypoints to a (25, 3) BODY_25 array."""
    body25_keypoints = np.zeros((25, 3), dtype=np.float32)

    for body25_index, coco18_index in _COCO18_FROM_BODY25.items():
        keypoint = (
            coco18_keypoints[coco18_index]
            if coco18_index < len(coco18_keypoints) else None)
        if keypoint is not None and keypoint.x >= 0 and keypoint.y >= 0:
            body25_keypoints[body25_index, 0] = keypoint.x * image_width
            body25_keypoints[body25_index, 1] = keypoint.y * image_height
            body25_keypoints[body25_index, 2] = (
                float(keypoint.score) if keypoint.score is not None else 1.0)

    # MidHip (index 8) = midpoint of RHip (COCO18[8]) and LHip (COCO18[11])
    right_hip = coco18_keypoints[8] if len(coco18_keypoints) > 8 else None
    left_hip = coco18_keypoints[11] if len(coco18_keypoints) > 11 else None
    if (right_hip is not None and left_hip is not None and
            right_hip.x >= 0 and right_hip.y >= 0 and
            left_hip.x >= 0 and left_hip.y >= 0):
        body25_keypoints[8, 0] = (right_hip.x + left_hip.x) * 0.5 * image_width
        body25_keypoints[8, 1] = (right_hip.y + left_hip.y) * 0.5 * image_height
        body25_keypoints[8, 2] = min(
            float(right_hip.score) if right_hip.score is not None else 1.0,
            float(left_hip.score) if left_hip.score is not None else 1.0,
        )

    return body25_keypoints


def _remap_hf_to_timm(hf_weights):
    """Convert rizvandwiki/gender-classification Transformers ViT weights to timm layout.

    The key difference is that HF stores Q, K, V as separate linear layers while
    timm merges them into a single QKV projection (shape [3*D, D]).
    """
    timm_state_dict = {}

    direct_key_map = {
        'classifier.weight':                              'head.weight',
        'classifier.bias':                                'head.bias',
        'vit.embeddings.cls_token':                       'cls_token',
        'vit.embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
        'vit.embeddings.patch_embeddings.projection.bias':   'patch_embed.proj.bias',
        'vit.embeddings.position_embeddings':             'pos_embed',
        'vit.layernorm.weight':                           'norm.weight',
        'vit.layernorm.bias':                             'norm.bias',
    }
    for huggingface_key, timm_key in direct_key_map.items():
        timm_state_dict[timm_key] = hf_weights[huggingface_key]

    for layer_index in range(12):
        huggingface_prefix = f'vit.encoder.layer.{layer_index}'
        timm_prefix = f'blocks.{layer_index}'

        timm_state_dict[f'{timm_prefix}.norm1.weight'] = (
            hf_weights[f'{huggingface_prefix}.layernorm_before.weight'])
        timm_state_dict[f'{timm_prefix}.norm1.bias'] = (
            hf_weights[f'{huggingface_prefix}.layernorm_before.bias'])

        query_weight = hf_weights[f'{huggingface_prefix}.attention.attention.query.weight']
        key_weight = hf_weights[f'{huggingface_prefix}.attention.attention.key.weight']
        value_weight = hf_weights[f'{huggingface_prefix}.attention.attention.value.weight']
        timm_state_dict[f'{timm_prefix}.attn.qkv.weight'] = torch.cat(
            [query_weight, key_weight, value_weight], dim=0)

        query_bias = hf_weights[f'{huggingface_prefix}.attention.attention.query.bias']
        key_bias = hf_weights[f'{huggingface_prefix}.attention.attention.key.bias']
        value_bias = hf_weights[f'{huggingface_prefix}.attention.attention.value.bias']
        timm_state_dict[f'{timm_prefix}.attn.qkv.bias'] = torch.cat(
            [query_bias, key_bias, value_bias], dim=0)

        timm_state_dict[f'{timm_prefix}.attn.proj.weight'] = (
            hf_weights[f'{huggingface_prefix}.attention.output.dense.weight'])
        timm_state_dict[f'{timm_prefix}.attn.proj.bias'] = (
            hf_weights[f'{huggingface_prefix}.attention.output.dense.bias'])

        timm_state_dict[f'{timm_prefix}.norm2.weight'] = (
            hf_weights[f'{huggingface_prefix}.layernorm_after.weight'])
        timm_state_dict[f'{timm_prefix}.norm2.bias'] = (
            hf_weights[f'{huggingface_prefix}.layernorm_after.bias'])

        timm_state_dict[f'{timm_prefix}.mlp.fc1.weight'] = (
            hf_weights[f'{huggingface_prefix}.intermediate.dense.weight'])
        timm_state_dict[f'{timm_prefix}.mlp.fc1.bias'] = (
            hf_weights[f'{huggingface_prefix}.intermediate.dense.bias'])

        timm_state_dict[f'{timm_prefix}.mlp.fc2.weight'] = (
            hf_weights[f'{huggingface_prefix}.output.dense.weight'])
        timm_state_dict[f'{timm_prefix}.mlp.fc2.bias'] = (
            hf_weights[f'{huggingface_prefix}.output.dense.bias'])

    return timm_state_dict


def _crop_person(image_rgb, body25_keypoints, padding_fraction=0.15):
    """Return a tight bounding-box crop of one person using their BODY_25 keypoints.

    body25_keypoints: (25, 3) pixel-coord array (x, y, confidence).
    Returns an H×W×3 uint8 RGB array, or None if fewer than 2 joints are visible.
    """
    visible = body25_keypoints[body25_keypoints[:, 2] > 0.1]
    if len(visible) < 2:
        return None
    image_height, image_width = image_rgb.shape[:2]
    min_x = visible[:, 0].min()
    min_y = visible[:, 1].min()
    max_x = visible[:, 0].max()
    max_y = visible[:, 1].max()
    padding_x = (max_x - min_x) * padding_fraction
    padding_y = (max_y - min_y) * padding_fraction
    min_x = max(0, int(min_x - padding_x))
    min_y = max(0, int(min_y - padding_y))
    max_x = min(image_width, int(max_x + padding_x))
    max_y = min(image_height, int(max_y + padding_y))
    if max_x <= min_x or max_y <= min_y:
        return None
    return image_rgb[min_y:max_y, min_x:max_x]


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

    def predict(self, image_rgb_uint8, body25_keypoints):
        """Return 'female' or 'male' for the person described by body25_keypoints.

        Falls back to 'neutral' when the crop is too small to be reliable.
        """
        crop = _crop_person(image_rgb_uint8, body25_keypoints)
        if crop is None:
            return 'neutral'

        # Pad to square before resizing to avoid aspect-ratio distortion
        crop_height, crop_width = crop.shape[:2]
        square_side = max(crop_height, crop_width)
        padded_crop = np.zeros((square_side, square_side, 3), dtype=np.uint8)
        padded_crop[
            (square_side - crop_height) // 2:
            (square_side - crop_height) // 2 + crop_height,
            (square_side - crop_width) // 2:
            (square_side - crop_width) // 2 + crop_width] = crop
        crop_image = Image.fromarray(padded_crop).resize((224, 224), Image.BILINEAR)
        image_tensor = TF.to_tensor(crop_image)                            # [0, 1]
        image_tensor = TF.normalize(
            image_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # [-1, 1]
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            predicted_label_index = self._model(image_tensor).argmax(dim=1).item()
        return self._LABELS[predicted_label_index]


_gender_clf = None   # loaded lazily on first call to detect_keypoints


class _Datum:
    """Minimal stand-in for pyopenpose's Datum; holds the detected arrays."""
    def __init__(self, pose_keypoints, left_hand_keypoints,
                 right_hand_keypoints, face_keypoints,
                 predicted_genders=None):
        self.poseKeypoints = pose_keypoints           # (N, 25, 3)
        self.handKeypoints = [left_hand_keypoints,    # (N, 21, 3)
                              right_hand_keypoints]   # (N, 21, 3)
        self.faceKeypoints = face_keypoints           # (N, 70, 3)
        self.gender_pd = predicted_genders or []      # list[str] length N


def detect_keypoints(image_bgr):
    """Detect all-person body, hand, face keypoints and predict gender.

    Returns a _Datum whose .poseKeypoints / .handKeypoints / .faceKeypoints
    match the shape conventions expected by pipeline.py / datum_to_keypoints(),
    and .gender_pd is a list[str] of 'female'/'male' predictions per person.
    """
    global _gender_clf

    image_height, image_width = image_bgr.shape[:2]

    poses = _detector.detect_poses(
        image_bgr,
        include_hand=True,
        include_face=True,
    )

    if not poses:
        empty = np.zeros((0, 1, 3), dtype=np.float32)
        return _Datum(empty, empty, empty, empty, predicted_genders=[])

    person_count = len(poses)
    pose_keypoints = np.zeros((person_count, 25, 3), dtype=np.float32)
    left_hand_keypoints = np.zeros((person_count, 21, 3), dtype=np.float32)
    right_hand_keypoints = np.zeros((person_count, 21, 3), dtype=np.float32)
    face_keypoints = np.zeros((person_count, 70, 3), dtype=np.float32)

    for person_index, pose in enumerate(poses):
        pose_keypoints[person_index] = _coco18_to_body25(
            pose.body.keypoints, image_height, image_width)
        if pose.left_hand is not None:
            left_hand_keypoints[person_index] = _keypoints_to_array(
                pose.left_hand, 21, image_height, image_width)
        if pose.right_hand is not None:
            right_hand_keypoints[person_index] = _keypoints_to_array(
                pose.right_hand, 21, image_height, image_width)
        if pose.face is not None:
            face_keypoints[person_index] = _keypoints_to_array(
                pose.face, 70, image_height, image_width)

    # Gender classification — load model once, then predict per person
    if _gender_clf is None:
        _gender_clf = _GenderClassifier()

    image_rgb = image_bgr[:, :, ::-1].copy()   # BGR → RGB uint8
    predicted_genders = [
        _gender_clf.predict(image_rgb, pose_keypoints[person_index])
        for person_index in range(person_count)]
    print(f'Gender predictions: {predicted_genders}')

    return _Datum(
        pose_keypoints, left_hand_keypoints, right_hand_keypoints,
        face_keypoints, predicted_genders=predicted_genders)
