import numpy as np
import numba
from numba import types
from numba.typed import List
from numba import jit, prange

@jit(nopython=True)
def mat4x4_identity():
    return np.eye(4, dtype=np.float32)

@jit(nopython=True, fastmath=True)
def projection_matrix(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg)/2)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = -(far + near)/(far - near)
    m[2,3] = -2*far*near/(far - near)
    m[3,2] = -1.0
    return m

@jit(nopython=True, fastmath=True)
def rotation_matrices(anglex, angley, anglez):
    rz = mat4x4_identity()
    rx = mat4x4_identity()
    ry = mat4x4_identity()
    rz[0,0] = np.cos(anglez)
    rz[0,1] = -np.sin(anglez)
    rz[1,0] = np.sin(anglez)
    rz[1,1] = np.cos(anglez)
    rx[1,1] = np.cos(anglex)
    rx[1,2] = -np.sin(anglex)
    rx[2,1] = np.sin(anglex)
    rx[2,2] = np.cos(anglex)
    ry[0,0] = np.cos(angley)
    ry[0,2] = np.sin(angley)
    ry[2,0] = -np.sin(angley)
    ry[2,2] = np.cos(angley)
    return rx, ry, rz

@jit(nopython=True, fastmath=True)
def transform(tri_vertices, mat, tri_h, tri_t):
    tri_h[:, :3] = tri_vertices
    tri_h[:, 3] = 1.0
    tri_t = (mat @ tri_h.T).T
    w = tri_t[:, 3]
    w[w == 0] = 1e-6
    tri_t[:, :3] /= w[:, np.newaxis]
    return tri_t[:, :3]

@jit(nopython=True, fastmath=True)
def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view

@jit(nopython=True, fastmath=True)
def calc_clip(tri_proj_clip, tri_view, proj_mat):
    for i in range(3):
        v4 = np.empty(4, dtype=np.float32)
        v4[:3] = tri_view[i]
        v4[3] = 1.0
        tri_proj_clip[i] = proj_mat @ v4

@jit(nopython=True, fastmath=True)
def calc_ndc(tri_proj_clip, tri_proj_ndc):
    for i in range(3):
        clip = tri_proj_clip[i]
        wv = clip[3] if clip[3] != 0.0 else 1e-6
        tri_proj_ndc[i, 0] = clip[0] / wv
        tri_proj_ndc[i, 1] = clip[1] / wv
        tri_proj_ndc[i, 2] = clip[2] / wv

@jit(nopython=True, fastmath=True)
def check_in_view(tri_proj_ndc):
    min_x = min(tri_proj_ndc[0, 0], tri_proj_ndc[1, 0], tri_proj_ndc[2, 0])
    max_x = max(tri_proj_ndc[0, 0], tri_proj_ndc[1, 0], tri_proj_ndc[2, 0])
    min_y = min(tri_proj_ndc[0, 1], tri_proj_ndc[1, 1], tri_proj_ndc[2, 1])
    max_y = max(tri_proj_ndc[0, 1], tri_proj_ndc[1, 1], tri_proj_ndc[2, 1])
    min_z = min(tri_proj_ndc[0, 2], tri_proj_ndc[1, 2], tri_proj_ndc[2, 2])
    max_z = max(tri_proj_ndc[0, 2], tri_proj_ndc[1, 2], tri_proj_ndc[2, 2])
    if max_x < -1.0 or min_x > 1.0 or max_y < -1.0 or min_y > 1.0 or max_z < -1.0 or min_z > 1.0: return False
    return True

@jit(nopython=True, fastmath=True)
def edge_function(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

@jit(nopython=True, fastmath=True)
def get_x_on_line(x1, y1, x2, y2, y):
    if y2 == y1: return x1
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

@jit(nopython=True, fastmath=True)
def point_to_screen(point_3d, view_mat, proj_mat, width, height, z_buffer):
    point_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float32)
    clip_pos = proj_mat @ view_mat @ point_h
    if clip_pos[3] <= 1e-6: return (0, 0, False)
    ndc_x = clip_pos[0] / clip_pos[3]
    ndc_y = clip_pos[1] / clip_pos[3]
    ndc_z = clip_pos[2] / clip_pos[3]
    if ndc_x < -1.0 or ndc_x > 1.0 or ndc_y < -1.0 or ndc_y > 1.0 or ndc_z < -1.0 or ndc_z > 1.0:return (0, 0, False)
    screen_x = int((ndc_x * 0.5 + 0.5) * (width - 1))
    screen_y = int((1.0 - (ndc_y * 0.5 + 0.5)) * (height - 1))
    visibility = True
    if 0 <= screen_y < height and 0 <= screen_x < width:
        scene_depth = z_buffer[screen_y, screen_x]
        bias = 0.001 
        if ndc_z > scene_depth + bias: visibility = False
    return (screen_x, screen_y, visibility)