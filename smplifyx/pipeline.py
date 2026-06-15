# -*- coding: utf-8 -*-
"""End-to-end pipeline: detect keypoints with OpenPose → fit SMPLify-X per
person → render all fitted meshes onto the original image.

Usage:
    python smplifyx/pipeline.py \\
        --config cfg_files/fit_smplx.yaml \\
        --image data_smplifyx/images/5ppl.jpg \\
        --output_folder smplx_output_5ppl \\
        --model_folder /Users/yasen/Documents/SherlockMaterials/dataset/smplx/models \\
        --vposer_ckpt vposer
"""

from __future__ import absolute_import, print_function, division

import sys
import os
import os.path as osp
import glob
import time
import yaml

import cv2
import numpy as np
import torch

import smplx

sys.path.insert(0, osp.dirname(__file__))
from utils import JointMapper, smpl_to_openpose
from cmd_parser import parse_config
from fit_single_frame import fit_single_frame
from camera import create_camera
from prior import create_prior
from render_multi_person import render_multi_person
import openpose_wrapper

torch.backends.cudnn.enabled = False

Keypoints_fields = ['keypoints', 'gender_gt', 'gender_pd']


def datum_to_keypoints(datum, use_hands=True, use_face=True,
                       use_face_contour=False):
    """Convert an OpenPose datum to the (N, J, 3) keypoints array that
    fit_single_frame expects, following the same logic as easy_run.py."""
    if datum.poseKeypoints is None or len(datum.poseKeypoints) == 0:
        return np.zeros((0, 1, 3), dtype=np.float32)

    keypoints = []
    for idx, body_pose in enumerate(datum.poseKeypoints):
        body_kp = np.array(body_pose, dtype=np.float32).reshape(-1, 3)

        if use_hands:
            lhand = np.array(datum.handKeypoints[0][idx],
                             dtype=np.float32).reshape(-1, 3)
            rhand = np.array(datum.handKeypoints[1][idx],
                             dtype=np.float32).reshape(-1, 3)
            body_kp = np.concatenate([body_kp, lhand, rhand], axis=0)

        if use_face:
            # 51 FLAME-compatible landmarks starting at offset 17
            face_kp = np.array(datum.faceKeypoints[idx],
                               dtype=np.float32).reshape(-1, 3)[17:68, :]
            contour = np.zeros((0, 3), dtype=np.float32)
            if use_face_contour:
                contour = np.array(datum.faceKeypoints[idx],
                                   dtype=np.float32).reshape(-1, 3)[:17, :]
            body_kp = np.concatenate([body_kp, face_kp, contour], axis=0)

        keypoints.append(body_kp)

    return np.stack(keypoints)   # (N_persons, J, 3)


def setup_smplx(args, dtype, device):
    """Instantiate body models, camera, and all priors — mirrors main.py."""
    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)
    model_type = args.get('model_type', 'smplx')

    joint_mapper = JointMapper(
        smpl_to_openpose(model_type,
                         use_hands=use_hands,
                         use_face=use_face,
                         use_face_contour=args.get('use_face_contour', False),
                         openpose_format=args.get('openpose_format', 'coco25')))

    # Build model_params without 'gender' so we can pass it explicitly per model
    args_no_gender = {k: v for k, v in args.items() if k != 'gender'}
    model_params = dict(
        model_path=args.get('model_folder'),
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not args.get('use_vposer'),
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=dtype,
        **args_no_gender)

    neutral_model = smplx.create(gender='neutral', **model_params)
    male_model = smplx.create(gender='male', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype, **args)
    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    body_pose_prior = create_prior(prior_type=args.get('body_prior_type'),
                                   dtype=dtype, **args)

    jaw_prior = expr_prior = None
    if use_face:
        jaw_prior = create_prior(prior_type=args.get('jaw_prior_type'),
                                 dtype=dtype, **args)
        expr_prior = create_prior(prior_type=args.get('expr_prior_type', 'l2'),
                                  dtype=dtype, **args)

    left_hand_prior = right_hand_prior = None
    if use_hands:
        lhand_args = {**args, 'num_gaussians': args.get('num_pca_comps')}
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype, use_left_hand=True, **lhand_args)
        rhand_args = {**args, 'num_gaussians': args.get('num_pca_comps')}
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype, use_right_hand=True, **rhand_args)

    shape_prior = create_prior(prior_type=args.get('shape_prior_type', 'l2'),
                               dtype=dtype, **args)
    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    # Joint weights: one per OpenPose joint
    num_joints = (25 + 2 * 20 * use_hands)
    optim_weights = np.ones(
        num_joints + 2 * use_hands + use_face * 51, dtype=np.float32)
    joints_to_ign = args.get('joints_to_ign')
    if joints_to_ign and -1 not in joints_to_ign:
        optim_weights[joints_to_ign] = 0.0
    joint_weights = torch.tensor(optim_weights, dtype=dtype, device=device)
    joint_weights = joint_weights.unsqueeze(0)

    # Move to device
    for obj in [camera, neutral_model, male_model, female_model,
                body_pose_prior, angle_prior, shape_prior]:
        obj.to(device=device)
    if use_face:
        expr_prior.to(device=device)
        jaw_prior.to(device=device)
    if use_hands:
        left_hand_prior.to(device=device)
        right_hand_prior.to(device=device)

    return dict(
        neutral_model=neutral_model,
        male_model=male_model,
        female_model=female_model,
        camera=camera,
        joint_weights=joint_weights,
        body_pose_prior=body_pose_prior,
        jaw_prior=jaw_prior,
        expr_prior=expr_prior,
        left_hand_prior=left_hand_prior,
        right_hand_prior=right_hand_prior,
        shape_prior=shape_prior,
        angle_prior=angle_prior,
    )


def main(**args):
    image_path = args.pop('image')
    output_folder = osp.expandvars(args.pop('output_folder'))
    max_persons = args.pop('max_persons', -1)

    os.makedirs(output_folder, exist_ok=True)
    result_folder = osp.join(output_folder, args.pop('result_folder', 'results'))
    mesh_folder = osp.join(output_folder, args.pop('mesh_folder', 'meshes'))
    img_out_folder = osp.join(output_folder, 'images')
    for d in [result_folder, mesh_folder, img_out_folder]:
        os.makedirs(d, exist_ok=True)

    # Save config
    with open(osp.join(output_folder, 'conf.yaml'), 'w') as f:
        yaml.dump(args, f)

    float_dtype = args.get('float_dtype', 'float32')
    dtype = torch.float64 if float_dtype == 'float64' else torch.float32
    device = torch.device('cpu')   # CPU-only on macOS

    # Force CPU-only when CUDA is unavailable (e.g., macOS)
    if not torch.cuda.is_available():
        if args.get('use_cuda', False):
            print('CUDA not available — switching to CPU.')
            args['use_cuda'] = False
        if args.get('interpenetration', False):
            print('CUDA not available — disabling interpenetration loss.')
            args['interpenetration'] = False

    # --- Detect keypoints ---
    print(f'Detecting keypoints in {image_path} ...')
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    img_rgb = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0

    datum = openpose_wrapper.detect_keypoints(img_bgr)

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)
    keypoints = datum_to_keypoints(datum, use_hands=use_hands,
                                   use_face=use_face)

    n_persons = keypoints.shape[0]
    print(f'Detected {n_persons} person(s)')
    if n_persons == 0:
        print('No people detected — exiting.')
        return

    if max_persons > 0:
        keypoints = keypoints[:max_persons]
        n_persons = keypoints.shape[0]

    # --- Set up SMPLify-X ---
    print('Setting up SMPLify-X models ...')
    setup = setup_smplx(args, dtype, device)

    # --- Fit each person ---
    result_paths = []
    for person_id in range(n_persons):
        print(f'\nFitting person {person_id} / {n_persons - 1} ...')
        result_fn = osp.join(result_folder, f'{person_id:03d}.pkl')
        mesh_fn = osp.join(mesh_folder, f'{person_id:03d}.obj')
        per_img_folder = osp.join(img_out_folder, f'{person_id:03d}')
        os.makedirs(per_img_folder, exist_ok=True)
        out_img_fn = osp.join(per_img_folder, 'output.png')

        fit_single_frame(
            img_rgb,
            keypoints[[person_id]],
            body_model=setup['neutral_model'],
            camera=setup['camera'],
            joint_weights=setup['joint_weights'],
            dtype=dtype,
            output_folder=output_folder,
            result_folder=result_folder,
            out_img_fn=out_img_fn,
            result_fn=result_fn,
            mesh_fn=mesh_fn,
            shape_prior=setup['shape_prior'],
            expr_prior=setup['expr_prior'],
            body_pose_prior=setup['body_pose_prior'],
            left_hand_prior=setup['left_hand_prior'],
            right_hand_prior=setup['right_hand_prior'],
            jaw_prior=setup['jaw_prior'],
            angle_prior=setup['angle_prior'],
            **args)

        if osp.exists(result_fn):
            result_paths.append(result_fn)

    # --- Composite render ---
    if result_paths:
        composite_path = osp.join(img_out_folder, 'composite.png')
        print(f'\nRendering composite of {len(result_paths)} person(s) ...')
        render_multi_person(result_paths, image_path, composite_path,
                            focal_length=float(args.get('focal_length', 5000)))
    else:
        print('No results saved — skipping composite render.')


if __name__ == '__main__':
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--image', required=True,
                      help='Input image path (single multi-person image)')
    _pre_args, _remaining = _pre.parse_known_args()
    args = parse_config(_remaining)
    args['image'] = _pre_args.image
    main(**args)
