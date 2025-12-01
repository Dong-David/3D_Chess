import numba
from numba import types
from numba.typed import List
from numba import objmode
from numba import jit, prange
import numpy as np
import os
from PIL import Image

from .geometry import *
from .math3d import *

thread_nums = min(6, os.cpu_count() - 1)

@jit(nopython=True, fastmath=True, nogil=True)
def process_mesh_mapping(mesh, pivot, pos, rot_mat, view_mat, proj_mat, width, height, start_idx, end_idx, 
                screen_pts, z_list, clipped_screen_pts, clipped_z_list):
    tri_cnt = 0; clipped_tri_cnt = 0
    MAX_CLIPPED = 64
    new_tri_clip = np.empty((3, 4), dtype=np.float32)
    temp_poly_array = np.zeros((MAX_CLIPPED, 4), dtype=np.float32)
    temp_uv_array = np.zeros((MAX_CLIPPED, 2), dtype=np.float32)
    tri_proj_clip = np.empty((3, 4), dtype=np.float32)
    tri_proj_ndc = np.empty((3, 3), dtype=np.float32)
    px_py = np.empty((3, 2), dtype=np.int32)
    z_vals = np.empty(3, dtype=np.float32)
    w_vals = np.empty(3, dtype=np.float32)
    tri_h = np.ones((3,4), np.float32)
    tri_t = np.zeros((3,4), np.float32)
    v0 = np.zeros(6, dtype=np.float32)
    v1 = np.zeros(6, dtype=np.float32)
    v2 = np.zeros(6, dtype=np.float32)
    buf_a = np.zeros((18, 6), dtype=np.float32)
    buf_b = np.zeros((18, 6), dtype=np.float32)
    pivot_mat = np.eye(4, dtype=np.float32)
    pivot_mat[:3, 3] = -pivot
    unpivot_mat = np.eye(4, dtype=np.float32)
    unpivot_mat[:3, 3] = pivot
    translation_mat = np.eye(4, dtype=np.float32)
    translation_mat[:3, 3] = pos
    model_mat = translation_mat @ unpivot_mat @ rot_mat @ pivot_mat
    mvp_mat = proj_mat @ view_mat @ model_mat
    mv_mat = view_mat @ model_mat
    
    for tri_idx in range(len(mesh)):
        if tri_idx < start_idx or tri_idx >= end_idx: continue
        tri = mesh[tri_idx]
        tri_view = transform(tri, mv_mat, tri_h, tri_t)

        edge1 = tri_view[1] - tri_view[0]
        edge2 = tri_view[2] - tri_view[0]
        n = np.cross(edge1, edge2)
        to_camera = -tri_view[0]  
        if np.dot(n, to_camera) <= 0: continue

        calc_clip(tri_proj_clip, tri_view, proj_mat)
        calc_ndc(tri_proj_clip, tri_proj_ndc)
        if calc_polygon(tri_proj_ndc, tri_proj_clip, px_py, z_vals, w_vals, width, height):
            screen_pts[tri_cnt, 0, 0] = px_py[0, 0]
            screen_pts[tri_cnt, 0, 1] = px_py[0, 1]
            screen_pts[tri_cnt, 1, 0] = px_py[1, 0]
            screen_pts[tri_cnt, 1, 1] = px_py[1, 1]
            screen_pts[tri_cnt, 2, 0] = px_py[2, 0]
            screen_pts[tri_cnt, 2, 1] = px_py[2, 1]
            
            z_list[tri_cnt, 0] = z_vals[0]
            z_list[tri_cnt, 1] = z_vals[1]
            z_list[tri_cnt, 2] = z_vals[2]
            tri_cnt += 1
        else:
            if not check_in_view(tri_proj_ndc): continue
            current_tri_uvs = np.zeros((3, 2), dtype=np.float32)
            num_new_vertices = triangulation_polygon(tri_proj_clip, current_tri_uvs, temp_poly_array, temp_uv_array, v0, v1, v2, buf_a, buf_b)
            num_new_triangles = num_new_vertices // 3
            for i in range(num_new_triangles):
                v0_idx, v1_idx, v2_idx = i * 3, i * 3 + 1, i * 3 + 2
                new_tri_clip[0] = temp_poly_array[v0_idx]
                new_tri_clip[1] = temp_poly_array[v1_idx]
                new_tri_clip[2] = temp_poly_array[v2_idx]
                calc_ndc(new_tri_clip, tri_proj_ndc)
                if calc_polygon(tri_proj_ndc, new_tri_clip, px_py, z_vals, w_vals, width, height):
                    clipped_screen_pts[clipped_tri_cnt, 0, 0] = px_py[0, 0]
                    clipped_screen_pts[clipped_tri_cnt, 0, 1] = px_py[0, 1]
                    clipped_screen_pts[clipped_tri_cnt, 1, 0] = px_py[1, 0]
                    clipped_screen_pts[clipped_tri_cnt, 1, 1] = px_py[1, 1]
                    clipped_screen_pts[clipped_tri_cnt, 2, 0] = px_py[2, 0]
                    clipped_screen_pts[clipped_tri_cnt, 2, 1] = px_py[2, 1]
                    
                    clipped_z_list[clipped_tri_cnt, 0] = z_vals[0]
                    clipped_z_list[clipped_tri_cnt, 1] = z_vals[1]
                    clipped_z_list[clipped_tri_cnt, 2] = z_vals[2]
                    
                    clipped_tri_cnt += 1
    return tri_cnt, clipped_tri_cnt

@jit(nopython=True, fastmath=True, nogil=True)
def rasterize(pts, z_vals, shadow_map, width, height):
    h, w = height, width
    x0 = float(pts[0,0]); y0 = float(pts[0,1])
    x1 = float(pts[1,0]); y1 = float(pts[1,1])
    x2 = float(pts[2,0]); y2 = float(pts[2,1])

    z0 = float(z_vals[0]); z1 = float(z_vals[1]); z2 = float(z_vals[2])
    if y1 < y0:
        x0, x1 = x1, x0; y0, y1 = y1, y0; z0, z1 = z1, z0
    if y2 < y0:
        x0, x2 = x2, x0; y0, y2 = y2, y0; z0, z2 = z2, z0
    if y2 < y1:
        x1, x2 = x2, x1; y1, y2 = y2, y1; z1, z2 = z2, z1

    min_y = max(int(np.floor(y0)), 0)
    max_y = min(int(np.ceil(y2)), h - 1)

    area = edge_function(x0, y0, x1, y1, x2, y2)
    if abs(area) < 1e-12:
        return  

    for y in range(min_y, max_y + 1):
        py = y + 0.5
        x_left = get_x_on_line(x0, y0, x2, y2, py)
        if py <= y1:
            x_right = get_x_on_line(x0, y0, x1, y1, py)
        else:
            x_right = get_x_on_line(x1, y1, x2, y2, py)

        if x_left > x_right:
            tmp = x_left; x_left = x_right; x_right = tmp

        min_x = max(int(np.floor(x_left)), 0)
        max_x = min(int(np.ceil(x_right)), w - 1)

        inv_area = 1.0 / area
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0_bary = edge_function(x1, y1, x2, y2, px, py)
            w1_bary = edge_function(x2, y2, x0, y0, px, py)
            w2_bary = edge_function(x0, y0, x1, y1, px, py)

            if (w0_bary >= 0.0 and w1_bary >= 0.0 and w2_bary >= 0.0) or (w0_bary <= 0.0 and w1_bary <= 0.0 and w2_bary <= 0.0):
                w0 = w0_bary * inv_area
                w1 = w1_bary * inv_area
                w2 = w2_bary * inv_area
                z = w0 * z0 + w1 * z1 + w2 * z2
                
                if z < shadow_map[y, x]:
                    shadow_map[y, x] = z


@jit(nopython=True, fastmath=True, parallel=True)
def create_shadow_map_parallel(mesh_list, pivot_list, pos_list, rot_list, view_mat, proj_mat, width, height, shadow_map):
    total_tris = 0
    NUM_BANDS = thread_nums
    for mesh in mesh_list: total_tris += len(mesh)
    max_tris = (total_tris + NUM_BANDS - 1) // NUM_BANDS + 100
    thread_counts = np.zeros(NUM_BANDS, dtype=np.int32)
    thread_pts = np.empty((NUM_BANDS, max_tris, 3, 2), dtype=np.int32)
    thread_z = np.empty((NUM_BANDS, max_tris, 3), dtype=np.float32)
    thread_clipped_screen_pts = np.empty((NUM_BANDS, max_tris, 3, 2), dtype=np.int32)
    thread_clipped_z_list = np.empty((NUM_BANDS, max_tris, 3), dtype=np.float32)

    for mesh_idx in range(len(mesh_list)):
        mesh = mesh_list[mesh_idx]
        pivot = pivot_list[mesh_idx]
        pos = pos_list[mesh_idx]
        irx, iry, irz = rot_list[mesh_idx]
        rx, ry, rz = rotation_matrices(irx, iry, irz)
        rot_mat = rz @ ry @ rx
        for band_index in prange(NUM_BANDS):
            band_size = (len(mesh) + NUM_BANDS - 1) // NUM_BANDS
            start_idx = band_index * band_size
            end_idx = min((band_index + 1) * band_size, len(mesh))

            screen_pts = thread_pts[band_index]
            z_list = thread_z[band_index]
            clipped_screen_pts = thread_clipped_screen_pts[band_index]
            clipped_z_list = thread_clipped_z_list[band_index]

            tri_cnt, clipped_tri_cnt = process_mesh_mapping(mesh, pivot, pos, rot_mat, view_mat, proj_mat, width, height, start_idx, end_idx, 
                    screen_pts, z_list, clipped_screen_pts, clipped_z_list)
            local_count = thread_counts[band_index]
            for j in range(tri_cnt):
                if local_count < max_tris:
                    pts = screen_pts[j]
                    z_vals = z_list[j]
                    for k in range(3):
                        thread_pts[band_index, local_count, k, 0] = pts[k][0]
                        thread_pts[band_index, local_count, k, 1] = pts[k][1]
                    thread_z[band_index, local_count, :] = z_vals
                    local_count += 1
            for j in range(clipped_tri_cnt):
                if local_count < max_tris:
                    pts = clipped_screen_pts[j]
                    z_vals = clipped_z_list[j]
                    for k in range(3):
                        thread_pts[band_index, local_count, k, 0] = pts[k][0]
                        thread_pts[band_index, local_count, k, 1] = pts[k][1]
                    
                    thread_z[band_index, local_count, :] = z_vals
                    local_count += 1
            
            thread_counts[band_index] = local_count

        for band_index in range(NUM_BANDS):
            tri_cnt = thread_counts[band_index]
            for tri_idx in range(tri_cnt):
                pts = thread_pts[band_index, tri_idx]
                z_vals = thread_z[band_index, tri_idx]
                rasterize(pts, z_vals, shadow_map, width, height)

@jit(nopython=True, fastmath=True)
def create_shadow_map_serial(mesh_list, pivot_list, pos_list, rot_list, view_mat, proj_mat, width, height, shadow_map):
    max_tris = 0
    for m in mesh_list: max_tris += len(m)
    screen_pts = np.empty((max_tris, 3, 2), dtype=np.int32)
    z_list = np.empty((max_tris, 3), dtype=np.float32)
    clipped_screen_pts = np.empty((max_tris, 3, 2), dtype=np.int32)
    clipped_z_list = np.empty((max_tris, 3), dtype=np.float32)
    for mesh_idx in range(len(mesh_list)):
        mesh = mesh_list[mesh_idx]
        pivot = pivot_list[mesh_idx]
        pos = pos_list[mesh_idx]
        irx, iry, irz = rot_list[mesh_idx]
        rx, ry, rz = rotation_matrices(irx, iry, irz)
        rot_mat = rz @ ry @ rx
        tri_cnt, clipped_tri_cnt = process_mesh_mapping(mesh, pivot, pos, rot_mat, view_mat, proj_mat, width, height, 0, len(mesh), 
                screen_pts, z_list, clipped_screen_pts, clipped_z_list)
        for j in range(tri_cnt):
            pts = screen_pts[j]
            z_vals = z_list[j]
            rasterize(pts, z_vals, shadow_map, width, height)
        for j in range(clipped_tri_cnt):
            pts = clipped_screen_pts[j]
            z_vals = clipped_z_list[j]
            rasterize(pts, z_vals, shadow_map, width, height)

@jit(nopython=True, fastmath=True)
def create_shadow_map(mesh_list, pivot_list, pos_list, rot_list, view_mat, proj_mat, width, height, shadow_map, mode):
    if mode: create_shadow_map_parallel(mesh_list, pivot_list, pos_list, rot_list, view_mat, proj_mat, width, height, shadow_map)
    else: create_shadow_map_serial(mesh_list, pivot_list, pos_list, rot_list, view_mat, proj_mat, width, height, shadow_map)

@jit(nopython=True, fastmath=True)
def check_shadow(pts, z_vals, w_vals, width, height, cam_view_mat, cam_proj_mat, light_view_proj, shadow_map):
    shadow = True
    result = np.zeros(3, dtype=np.uint8)  
    cam_view_proj = cam_proj_mat @ cam_view_mat
    inv_cam_view_proj = np.linalg.inv(cam_view_proj.astype(np.float64)).astype(np.float32)  
    
    for i in range(3):
        ndc_x = (2.0 * pts[i, 0] / width) - 1.0
        ndc_y = 1.0 - (2.0 * pts[i, 1] / height)
        ndc_z = z_vals[i]
        clip_pos = np.array([ndc_x * w_vals[i], ndc_y * w_vals[i], ndc_z * w_vals[i], w_vals[i]], dtype=np.float32) 
        world_pos_h = inv_cam_view_proj @ clip_pos
        world_pos = world_pos_h[:3] / world_pos_h[3]
        world_pos_homo = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)  
        light_clip = light_view_proj @ world_pos_homo
        if light_clip[3] <= 1e-6:
            result[i] = 0
            continue

        light_ndc = light_clip[:3] / light_clip[3]
        if (light_ndc[0] < -1.0 or light_ndc[0] > 1.0 or 
            light_ndc[1] < -1.0 or light_ndc[1] > 1.0 or
            light_ndc[2] < -1.0 or light_ndc[2] > 1.0):
            result[i] = 0
            continue
        u = 0.5 * light_ndc[0] + 0.5
        v = 0.5 * light_ndc[1] + 0.5
        depth = 0.5 * light_ndc[2] + 0.5
        shadow_width = shadow_map.shape[1]
        shadow_height = shadow_map.shape[0]
        iu = int(u * (shadow_width - 1))
        iv = int(v * (shadow_height - 1))
        
        if iu < 0 or iu >= shadow_width or iv < 0 or iv >= shadow_height: result[i] = 0
        else:
            bias = 0.005
            if depth > shadow_map[iv, iu] + bias: result[i] = 1  
            else: result[i] = 0  
    if not result[0] and not result[1] and not result[2]: shadow = False
                
    return shadow

def visualize_shadow_map(shadow_map, save_path=None, show=True):
    shadow = np.nan_to_num(shadow_map, nan=1.0, posinf=1.0, neginf=-1.0)
    shadow = np.clip(shadow, -1.0, 1.0)
    img_data = (shadow + 1.0) * 0.5
    img = (img_data * 255).astype(np.uint8)
    img = Image.fromarray(img, mode='L')

    if save_path:
        img.save(save_path)
        print(f"✅ Saved shadow map to {save_path}")
    if show: img.show()
    return img