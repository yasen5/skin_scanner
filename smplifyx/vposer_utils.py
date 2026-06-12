# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import ast
import configparser
import glob
import importlib.util
import os
import os.path as osp
import sys


def _add_vposer_source_to_path(expr_dir):
    expr_dir = osp.abspath(osp.expandvars(expr_dir))
    candidates = [
        expr_dir,
        osp.dirname(expr_dir),
        osp.join(osp.dirname(expr_dir), 'human_body_prior'),
        osp.join(os.getcwd(), 'human_body_prior'),
    ]
    for candidate in candidates:
        if osp.isdir(osp.join(candidate, 'human_body_prior', 'tools')):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return


def _load_vposer_v1(expr_dir):
    import torch

    ckpts = sorted(glob.glob(osp.join(expr_dir, 'snapshots', '*.pt')),
                   key=osp.getmtime)
    if not ckpts:
        raise FileNotFoundError(
            'No VPoser v1 checkpoint found at {}'.format(
                osp.join(expr_dir, 'snapshots', '*.pt')))

    model_code = osp.join(expr_dir, 'vposer_smpl.py')
    if not osp.exists(model_code):
        raise FileNotFoundError(
            'No VPoser v1 model definition found at {}'.format(model_code))

    config_fns = sorted(glob.glob(osp.join(expr_dir, '*.ini')))
    if not config_fns:
        raise FileNotFoundError(
            'No VPoser v1 config .ini found in {}'.format(expr_dir))

    parser = configparser.ConfigParser()
    parser.read(config_fns[0])
    cfg = parser['All']

    spec = importlib.util.spec_from_file_location('vposer_smpl', model_code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.VPoser(
        num_neurons=cfg.getint('num_neurons'),
        latentD=cfg.getint('latentD'),
        data_shape=ast.literal_eval(cfg.get('data_shape')),
        use_cont_repr=cfg.getboolean('use_cont_repr'))

    state_dict = torch.load(ckpts[-1], map_location=torch.device('cpu'))
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, cfg


def load_vposer_model(expr_dir):
    expr_dir = osp.abspath(osp.expandvars(expr_dir))
    _add_vposer_source_to_path(expr_dir)

    if glob.glob(osp.join(expr_dir, 'snapshots', '*.pt')):
        return _load_vposer_v1(expr_dir)

    try:
        from human_body_prior.tools.model_loader import load_vposer
    except ImportError:
        from human_body_prior.models.vposer_model import VPoser
        from human_body_prior.tools.model_loader import load_model
        return load_model(
            expr_dir,
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
    return load_vposer(expr_dir, vp_model='snapshot')


def _rotation_matrix_to_angle_axis(rotation_matrix):
    import torch

    rot = rotation_matrix.reshape(-1, 3, 3)
    cos_angle = ((rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2] - 1.0) *
                 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)

    axis = torch.stack([
        rot[:, 2, 1] - rot[:, 1, 2],
        rot[:, 0, 2] - rot[:, 2, 0],
        rot[:, 1, 0] - rot[:, 0, 1],
    ], dim=1)

    scale = angle / (2.0 * sin_angle)
    small_angles = sin_angle.abs() < 1e-6
    scale = torch.where(small_angles, torch.full_like(scale, 0.5), scale)
    return axis * scale.unsqueeze(1)


def decode_vposer(vposer, pose_embedding):
    try:
        return vposer.decode(pose_embedding, output_type='aa').view(1, -1)
    except (RuntimeError, TypeError):
        decoded = vposer.decode(pose_embedding)
        if isinstance(decoded, dict):
            decoded = decoded['pose_body']
        if decoded.shape[-1] == 3:
            return decoded.reshape(pose_embedding.shape[0], -1)
        decoded = decoded.reshape(decoded.shape[0], -1, 3, 3)
        return _rotation_matrix_to_angle_axis(decoded).reshape(
            pose_embedding.shape[0], -1)
