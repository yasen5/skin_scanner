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
from PIL import Image, ImageDraw, ImageOps

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
    v = vertices.copy()
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    return v


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


def draw_edges(draw, projected, edges, color, width, image_size):
    image_width, image_height = image_size
    for start, end in edges:
        p0 = projected[start]
        p1 = projected[end]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        if ((p0[0] < -image_width and p1[0] < -image_width) or
                (p0[0] > image_width * 2 and p1[0] > image_width * 2) or
                (p0[1] < -image_height and p1[1] < -image_height) or
                (p0[1] > image_height * 2 and p1[1] > image_height * 2)):
            continue
        draw.line((float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])),
                  fill=color, width=width)


def render_multi_person(result_paths, image_path, output_path,
                        focal_length=5000.0):
    """Overlay each person's fitted mesh onto the image with a distinct colour."""
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, pkl_path in enumerate(sorted(result_paths)):
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f, encoding='latin1')

        mesh_path = infer_mesh_path(pkl_path)
        if not osp.exists(mesh_path):
            print(f'Mesh not found for person {i}: {mesh_path}')
            continue

        mesh = trimesh.load(mesh_path, process=False)
        vertices = exported_obj_to_model_space(np.asarray(mesh.vertices))
        faces = np.asarray(mesh.faces)

        R = np.asarray(result['camera_rotation']).reshape(3, 3)
        t = np.asarray(result['camera_translation']).reshape(3)

        projected, cam_verts = project_vertices(
            vertices, R, t, focal_length, image.size)

        c = PERSON_COLORS[i % len(PERSON_COLORS)]
        draw_edges(draw, projected, mesh_edges(faces),
                   color=(*c, 75), width=1, image_size=image.size)
        draw_edges(draw, projected, silhouette_edges(faces, cam_verts),
                   color=(*c, 230), width=2, image_size=image.size)
        print(f'Rendered person {i} ({osp.basename(pkl_path)}) in color {c}')

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

    pkl_files = sorted(glob.glob(osp.join(args.results_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f'No .pkl files found in {args.results_dir}')

    render_multi_person(pkl_files, args.image, args.out, args.focal_length)
