# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
# AMASS: Archive of Motion Capture as Surface Shapes <https://arxiv.org/abs/1904.03278>
#
#
# Code Developed by:
# Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# Vassilis Choutas <https://ps.is.tuebingen.mpg.de/employees/vchoutas> for ContinousRotReprDecoder
#
# 2018.01.02

'''
A human body pose prior built with Auto-Encoding Variational Bayes
'''

__all__ = ['VPoser']

import os
import shutil
import sys

import torch

from torch import nn
from torch.nn import functional as F

import numpy as np

import torchgeometry as tgm

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        first_basis_vector = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_product = torch.sum(
            first_basis_vector * reshaped_input[:, :, 1], dim=1, keepdim=True)
        second_basis_vector = F.normalize(
            reshaped_input[:, :, 1] - dot_product * first_basis_vector, dim=-1)
        third_basis_vector = torch.cross(
            first_basis_vector, second_basis_vector, dim=1)

        return torch.stack(
            [first_basis_vector, second_basis_vector, third_basis_vector],
            dim=-1)


class VPoser(nn.Module):
    def __init__(self, neuron_count=None, latent_dimension=None,
                 data_shape=None, use_continuous_representation=None,
                 **legacy_kwargs):
        super(VPoser, self).__init__()

        if neuron_count is None:
            neuron_count = legacy_kwargs.pop('num_neurons')
        if latent_dimension is None:
            latent_dimension = legacy_kwargs.pop('latentD')
        if data_shape is None:
            data_shape = legacy_kwargs.pop('data_shape')
        if use_continuous_representation is None:
            use_continuous_representation = legacy_kwargs.pop(
                'use_cont_repr', True)

        self.latent_dimension = latent_dimension
        self.use_continuous_representation = use_continuous_representation

        feature_count = np.prod(data_shape)
        self.joint_count = data_shape[1]

        self.bodyprior_enc_bn1 = nn.BatchNorm1d(feature_count)
        self.bodyprior_enc_fc1 = nn.Linear(feature_count, neuron_count)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(neuron_count)
        self.bodyprior_enc_fc2 = nn.Linear(neuron_count, neuron_count)
        self.bodyprior_enc_mu = nn.Linear(neuron_count, latent_dimension)
        self.bodyprior_enc_logvar = nn.Linear(neuron_count, latent_dimension)
        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latent_dimension, neuron_count)
        self.bodyprior_dec_fc2 = nn.Linear(neuron_count, neuron_count)

        if self.use_continuous_representation:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(neuron_count, self.joint_count * 6)

    def encode(self, pose_input):
        '''

        :param pose_input: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        encoded_pose = pose_input.view(pose_input.size(0), -1)  # flatten input
        encoded_pose = self.bodyprior_enc_bn1(encoded_pose)

        encoded_pose = F.leaky_relu(
            self.bodyprior_enc_fc1(encoded_pose), negative_slope=.2)
        encoded_pose = self.bodyprior_enc_bn2(encoded_pose)
        encoded_pose = self.dropout(encoded_pose)
        encoded_pose = F.leaky_relu(
            self.bodyprior_enc_fc2(encoded_pose), negative_slope=.2)
        return torch.distributions.normal.Normal(
            self.bodyprior_enc_mu(encoded_pose),
            F.softplus(self.bodyprior_enc_logvar(encoded_pose)))

    def decode(self, latent_input, output_type='matrot'):
        assert output_type in ['matrot', 'aa']

        decoded_pose = F.leaky_relu(
            self.bodyprior_dec_fc1(latent_input), negative_slope=.2)
        decoded_pose = self.dropout(decoded_pose)
        decoded_pose = F.leaky_relu(
            self.bodyprior_dec_fc2(decoded_pose), negative_slope=.2)
        decoded_pose = self.bodyprior_dec_out(decoded_pose)
        if self.use_continuous_representation:
            decoded_pose = self.rot_decoder(decoded_pose)
        else:
            decoded_pose = torch.tanh(decoded_pose)

        decoded_pose = decoded_pose.view([-1, 1, self.joint_count, 9])
        if output_type == 'aa':
            return VPoser.matrot2aa(decoded_pose)
        return decoded_pose

    def forward(self, pose_input, input_type='matrot', output_type='matrot'):
        '''

        :param pose_input: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        # if input_type == 'aa': pose_input = VPoser.aa2matrot(pose_input)
        latent_distribution = self.encode(pose_input)
        sampled_latent = latent_distribution.rsample()
        reconstructed_pose = self.decode(sampled_latent)
        if output_type == 'aa':
            reconstructed_pose = VPoser.matrot2aa(reconstructed_pose)

        # return reconstructed_pose, latent_distribution.mean, latent_distribution.sigma
        return {
            'pose': reconstructed_pose,
            'mean': latent_distribution.mean,
            'std': latent_distribution.scale}

    def sample_poses(self, pose_count=None, output_type='aa', seed=None,
                     **legacy_kwargs):
        if pose_count is None:
            pose_count = legacy_kwargs.pop('num_poses')
        np.random.seed(seed)
        dtype = self.bodyprior_dec_fc1.weight.dtype
        device = self.bodyprior_dec_fc1.weight.device
        self.eval()
        with torch.no_grad():
            generated_latent = torch.tensor(
                np.random.normal(0., 1., size=(pose_count, self.latent_dimension)),
                dtype=dtype).to(device)
        return self.decode(generated_latent, output_type=output_type)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogeneous_matrix_rotation = F.pad(pose_matrot.view(-1, 3, 3), [0, 1])
        pose = tgm.rotation_matrix_to_angle_axis(
            homogeneous_matrix_rotation).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        body_pose_matrix_rotation = tgm.angle_axis_to_rotation_matrix(
            pose.reshape(-1, 3))[:, :3, :3].contiguous().view(
                batch_size, 1, -1, 9)
        return body_pose_matrix_rotation
