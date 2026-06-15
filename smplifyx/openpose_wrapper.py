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
    Undetected keypoints (None) are returned with confidence 0.
    """
    arr = np.zeros((n_joints, 3), dtype=np.float32)
    for i, kp in enumerate(kp_list):
        if i >= n_joints:
            break
        if kp is not None:
            arr[i, 0] = kp.x * W
            arr[i, 1] = kp.y * H
            arr[i, 2] = float(kp.score) if kp.score is not None else 1.0
    return arr


def _coco18_to_body25(coco18, H, W):
    """Convert a list of 18 COCO Keypoints to a (25, 3) BODY_25 array."""
    body25 = np.zeros((25, 3), dtype=np.float32)

    for b25_idx, c18_idx in _COCO18_FROM_BODY25.items():
        kp = coco18[c18_idx] if c18_idx < len(coco18) else None
        if kp is not None:
            body25[b25_idx, 0] = kp.x * W
            body25[b25_idx, 1] = kp.y * H
            body25[b25_idx, 2] = float(kp.score) if kp.score is not None else 1.0

    # MidHip (index 8) = midpoint of RHip (COCO18[8]) and LHip (COCO18[11])
    r_hip = coco18[8] if len(coco18) > 8 else None
    l_hip = coco18[11] if len(coco18) > 11 else None
    if r_hip is not None and l_hip is not None:
        body25[8, 0] = (r_hip.x + l_hip.x) * 0.5 * W
        body25[8, 1] = (r_hip.y + l_hip.y) * 0.5 * H
        body25[8, 2] = min(
            float(r_hip.score) if r_hip.score is not None else 1.0,
            float(l_hip.score) if l_hip.score is not None else 1.0,
        )

    return body25


class _Datum:
    """Minimal stand-in for pyopenpose's Datum; holds the detected arrays."""
    def __init__(self, pose_kps, hand_kps_left, hand_kps_right, face_kps):
        self.poseKeypoints = pose_kps          # (N, 25, 3)
        self.handKeypoints = [hand_kps_left,   # (N, 21, 3)
                              hand_kps_right]  # (N, 21, 3)
        self.faceKeypoints = face_kps          # (N, 70, 3)


def detect_keypoints(image_bgr):
    """Detect all-person body, hand and face keypoints in a BGR numpy image.

    Returns a _Datum whose .poseKeypoints / .handKeypoints / .faceKeypoints
    match the shape conventions expected by pipeline.py / datum_to_keypoints().
    """
    H, W = image_bgr.shape[:2]

    poses = _detector.detect_poses(
        image_bgr,
        include_hand=True,
        include_face=True,
    )

    if not poses:
        empty = np.zeros((0, 1, 3), dtype=np.float32)
        return _Datum(empty, empty, empty, empty)

    n = len(poses)
    pose_kps = np.zeros((n, 25, 3), dtype=np.float32)
    lhand_kps = np.zeros((n, 21, 3), dtype=np.float32)
    rhand_kps = np.zeros((n, 21, 3), dtype=np.float32)
    face_kps = np.zeros((n, 70, 3), dtype=np.float32)

    for i, pose in enumerate(poses):
        pose_kps[i] = _coco18_to_body25(pose.body.keypoints, H, W)

        if pose.left_hand is not None:
            lhand_kps[i] = _keypoints_to_array(pose.left_hand, 21, H, W)
        if pose.right_hand is not None:
            rhand_kps[i] = _keypoints_to_array(pose.right_hand, 21, H, W)
        if pose.face is not None:
            face_kps[i] = _keypoints_to_array(pose.face, 70, H, W)

    return _Datum(pose_kps, lhand_kps, rhand_kps, face_kps)
