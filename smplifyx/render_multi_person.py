# -*- coding: utf-8 -*-
"""Render fitted SMPLify-X meshes for all persons onto a single image.

Usage:
    python smplifyx/render_multi_person.py \\
        --results_dir smplx_output/results/5ppl/ \\
        --image data_smplifyx/images/5ppl.jpg \\
        --out smplx_output/images/5ppl_composite.png
"""

import argparse
import glob
import os
import os.path as osp
import pickle

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Distinct colours per person (RGB, 0-255)
PERSON_COLORS = [
    (50, 220, 255),   # cyan
    (255, 110, 50),   # orange
    (100, 255, 80),   # green
    (220, 80, 255),   # purple
    (255, 220, 50),   # yellow
    (255, 80, 140),   # pink
    (80, 180, 255),   # blue
    (200, 255, 100),  # lime
]


def infer_mesh_path(result_path):
    mesh_path = result_path.replace(osp.sep + 'results' + osp.sep,
                                    osp.sep + 'meshes' + osp.sep)
    return osp.splitext(mesh_path)[0] + '.obj'


def exported_obj_to_model_space(vertices):
    """SMPLify-X exports OBJ meshes after a 180-degree rotation around X."""
    model_vertices = vertices.copy()
    model_vertices[:, 1] *= -1.0
    model_vertices[:, 2] *= -1.0
    return model_vertices


def project_vertices(vertices, camera_rotation, camera_translation,
                     focal_length, image_size):
    width, height = image_size
    camera_vertices = vertices.dot(camera_rotation.T) + camera_translation
    z = camera_vertices[:, 2]
    projected = np.empty((vertices.shape[0], 2), dtype=np.float64)
    projected[:, 0] = focal_length * camera_vertices[:, 0] / z + width * 0.5
    projected[:, 1] = focal_length * camera_vertices[:, 1] / z + height * 0.5
    return projected, camera_vertices


def mesh_edges(faces):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def silhouette_edges(faces, camera_vertices):
    face_vertices = camera_vertices[faces]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0],
                       face_vertices[:, 2] - face_vertices[:, 0])
    centers = face_vertices.mean(axis=1)
    front_facing = np.einsum('ij,ij->i', normals, centers) < 0

    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted(edge))
            edge_to_faces.setdefault(edge, []).append(face_idx)

    outline = []
    for edge, adjacent in edge_to_faces.items():
        if len(adjacent) == 1:
            outline.append(edge)
        elif front_facing[adjacent[0]] != front_facing[adjacent[1]]:
            outline.append(edge)
    return np.asarray(outline, dtype=np.int64)


def draw_edges(draw, projected_vertices, edges, color, width, image_size):
    image_width, image_height = image_size
    for start, end in edges:
        start_point = projected_vertices[start]
        end_point = projected_vertices[end]
        if not np.isfinite(start_point).all() or not np.isfinite(end_point).all():
            continue
        if ((start_point[0] < -image_width and end_point[0] < -image_width) or
                (start_point[0] > image_width * 2 and
                 end_point[0] > image_width * 2) or
                (start_point[1] < -image_height and
                 end_point[1] < -image_height) or
                (start_point[1] > image_height * 2 and
                 end_point[1] > image_height * 2)):
            continue
        draw.line(
            (float(start_point[0]), float(start_point[1]),
             float(end_point[0]), float(end_point[1])),
            fill=color, width=width)


def _load_font(size=16):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# Radii for body / hand / face keypoints
_KEYPOINT_RADIUS = {
    'body':  5,
    'hand':  3,
    'face':  2,
}
_KEYPOINT_CONFIDENCE_THRESHOLD = 0.1

# OpenPose COCO-25 body joint count
_BODY_JOINT_COUNT = 25
_HAND_JOINT_COUNT = 21   # per hand


def _draw_keypoints(draw, keypoints, color, image_size):
    """Draw all keypoints for one person.

    keypoints: (J, 3) array  —  (x, y, confidence)
    Layout: 0-24 body, 25-45 left hand, 46-66 right hand, 67+ face
    """
    for joint_index, keypoint in enumerate(keypoints):
        x_coordinate, y_coordinate, confidence = keypoint
        if confidence < _KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        if joint_index < _BODY_JOINT_COUNT:
            radius = _KEYPOINT_RADIUS['body']
        elif joint_index < _BODY_JOINT_COUNT + 2 * _HAND_JOINT_COUNT:
            radius = _KEYPOINT_RADIUS['hand']
        else:
            radius = _KEYPOINT_RADIUS['face']
        if not (np.isfinite(x_coordinate) and np.isfinite(y_coordinate)):
            continue
        draw.ellipse(
            [(x_coordinate - radius, y_coordinate - radius),
             (x_coordinate + radius, y_coordinate + radius)],
            fill=(*color, 220),
            outline=(*color, 255),
        )


def _label_position(keypoints):
    """Return (x, y) for the gender label — above the visible body keypoints."""
    body_keypoints = keypoints[:_BODY_JOINT_COUNT]
    visible_keypoints = body_keypoints[
        body_keypoints[:, 2] >= _KEYPOINT_CONFIDENCE_THRESHOLD]
    if len(visible_keypoints) == 0:
        visible_keypoints = keypoints[
            keypoints[:, 2] >= _KEYPOINT_CONFIDENCE_THRESHOLD]
    if len(visible_keypoints) == 0:
        return None
    # Nose or neck preferred (joints 0 and 1) for a natural label position
    for joint_index in (0, 1, 15, 16):
        if (joint_index < len(keypoints) and
                keypoints[joint_index, 2] >= _KEYPOINT_CONFIDENCE_THRESHOLD):
            return (
                float(keypoints[joint_index, 0]),
                float(keypoints[joint_index, 1]) - 18)
    return (
        float(visible_keypoints[:, 0].mean()),
        float(visible_keypoints[:, 1].min()) - 18)


def render_multi_person(result_paths, image_path, output_path,
                        focal_length=5000.0,
                        keypoints_per_person=None,
                        genders=None):
    """Overlay each person's fitted mesh, keypoints, and gender label."""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(size=16)

    for person_index, result_path in enumerate(sorted(result_paths)):
        with open(result_path, 'rb') as result_file:
            result = pickle.load(result_file, encoding='latin1')

        mesh_path = infer_mesh_path(result_path)
        person_color = PERSON_COLORS[person_index % len(PERSON_COLORS)]

        if not osp.exists(mesh_path):
            print(f'Mesh not found for person {person_index}: {mesh_path}')
        else:
            mesh = trimesh.load(mesh_path, process=False)
            vertices = exported_obj_to_model_space(np.asarray(mesh.vertices))
            faces = np.asarray(mesh.faces)

            camera_rotation = np.asarray(result['camera_rotation']).reshape(3, 3)
            camera_translation = np.asarray(result['camera_translation']).reshape(3)

            projected_vertices, camera_vertices = project_vertices(
                vertices, camera_rotation, camera_translation,
                focal_length, image.size)

            draw_edges(draw, projected_vertices, mesh_edges(faces),
                       color=(*person_color, 75), width=1, image_size=image.size)
            draw_edges(draw, projected_vertices,
                       silhouette_edges(faces, camera_vertices),
                       color=(*person_color, 230), width=2, image_size=image.size)

        # Draw OpenPose keypoints
        if (keypoints_per_person is not None and
                person_index < len(keypoints_per_person)):
            keypoints = np.asarray(keypoints_per_person[person_index])   # (J, 3)
            _draw_keypoints(draw, keypoints, person_color, image.size)

            # Draw gender label
            gender = (
                genders[person_index]
                if (genders is not None and person_index < len(genders))
                else 'neutral')
            label_position = _label_position(keypoints)
            if label_position is not None:
                x_coordinate, y_coordinate = label_position
                label = gender
                # Dark shadow for readability
                draw.text((x_coordinate + 1, y_coordinate + 1), label, font=font,
                          fill=(0, 0, 0, 200))
                draw.text((x_coordinate, y_coordinate), label, font=font,
                          fill=(*person_color, 255))

        print(
            f'Rendered person {person_index} ({osp.basename(result_path)}) '
            f'in color {person_color}')

    output = Image.alpha_composite(image, overlay).convert('RGB')
    os.makedirs(osp.dirname(output_path) or '.', exist_ok=True)
    output.save(output_path)
    print(f'Saved composite to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing per-person .pkl result files')
    parser.add_argument('--image', required=True,
                        help='Original input image')
    parser.add_argument('--out', required=True,
                        help='Output composite image path')
    parser.add_argument('--focal_length', type=float, default=5000.0)
    args = parser.parse_args()

    result_files = sorted(glob.glob(osp.join(args.results_dir, '*.pkl')))
    if not result_files:
        raise FileNotFoundError(f'No .pkl files found in {args.results_dir}')

    render_multi_person(result_files, args.image, args.out, args.focal_length)
