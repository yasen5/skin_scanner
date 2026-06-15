# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import pickle

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps

from cmd_parser import parse_config


def infer_mesh_path(result_path):
    mesh_path = result_path.replace(osp.sep + 'results' + osp.sep,
                                    osp.sep + 'meshes' + osp.sep)
    return osp.splitext(mesh_path)[0] + '.obj'


def exported_obj_to_model_space(vertices):
    """SMPLify-X exports OBJ meshes after a 180 degree rotation around X."""
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
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True,
                        help='SMPLify-X result .pkl file with camera parameters')
    parser.add_argument('--mesh',
                        help='Fitted SMPLify-X .obj mesh. If omitted, inferred from the result path.')
    parser.add_argument('--image', required=True,
                        help='Original image used for fitting')
    parser.add_argument('--out', required=True,
                        help='Output overlay image path')
    parser.add_argument('--draw', choices=['silhouette', 'wire', 'both'],
                        default='both',
                        help='Which projected mesh lines to draw')
    args, remaining = parser.parse_known_args()

    config = parse_config(remaining)
    focal_length = float(config.get('focal_length', 5000.0))

    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGBA')
    image_size = image.size

    with open(args.result, 'rb') as result_file:
        result = pickle.load(result_file, encoding='latin1')

    mesh_path = args.mesh or infer_mesh_path(args.result)
    mesh = trimesh.load(mesh_path, process=False)
    vertices = exported_obj_to_model_space(np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces)

    camera_rotation = np.asarray(result['camera_rotation']).reshape(3, 3)
    camera_translation = np.asarray(result['camera_translation']).reshape(3)
    projected, camera_vertices = project_vertices(
        vertices, camera_rotation, camera_translation, focal_length, image_size)

    overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if args.draw in ('wire', 'both'):
        draw_edges(draw, projected, mesh_edges(faces),
                   color=(50, 220, 255, 90), width=1, image_size=image_size)
    if args.draw in ('silhouette', 'both'):
        draw_edges(draw, projected, silhouette_edges(faces, camera_vertices),
                   color=(255, 230, 80, 240), width=2, image_size=image_size)

    output = Image.alpha_composite(image, overlay).convert('RGB')
    os.makedirs(osp.dirname(args.out) or '.', exist_ok=True)
    output.save(args.out)

    finite = np.isfinite(projected).all(axis=1)
    bounding_box = np.vstack([projected[finite].min(axis=0),
                              projected[finite].max(axis=0)])
    print('Wrote {}'.format(args.out))
    print('Projected mesh bounding box: [{:.1f}, {:.1f}] to [{:.1f}, {:.1f}]'.format(
        bounding_box[0, 0], bounding_box[0, 1],
        bounding_box[1, 0], bounding_box[1, 1]))
