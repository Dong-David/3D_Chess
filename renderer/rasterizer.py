from numba import jit, prange
import numpy as np

from .geometry import *
from .math3d import *
from .shadow_mapping import check_shadow
from .setting import *

@jit(nopython=True, fastmath=True)
def process_mesh(mesh, uv, pivot, pos, rot_mat, view_mat, proj_mat, width, height, start_idx, end_idx, 
                screen_pts, z_list, w_list, uvs_list, tri_indices):
    tri_cnt = 0;
    new_tri_clip = np.empty((3, 4), dtype=np.float32)
    new_tri_uv = np.empty((3, 2), dtype=np.float32)
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
            
            w_list[tri_cnt, 0] = w_vals[0]
            w_list[tri_cnt, 1] = w_vals[1]
            w_list[tri_cnt, 2] = w_vals[2]

            uvs_list[tri_cnt, 0, 0] = uv[tri_idx][0, 0]
            uvs_list[tri_cnt, 0, 1] = uv[tri_idx][0, 1]
            uvs_list[tri_cnt, 1, 0] = uv[tri_idx][1, 0]
            uvs_list[tri_cnt, 1, 1] = uv[tri_idx][1, 1]
            uvs_list[tri_cnt, 2, 0] = uv[tri_idx][2, 0]
            uvs_list[tri_cnt, 2, 1] = uv[tri_idx][2, 1]

            tri_indices[tri_cnt] = tri_idx
            tri_cnt += 1            
        else:
            if not check_in_view(tri_proj_ndc): continue
            current_tri_uvs = uv[tri_idx]
            num_new_vertices = triangulation_polygon(tri_proj_clip, current_tri_uvs, temp_poly_array, temp_uv_array, v0, v1, v2, buf_a, buf_b)
            num_new_triangles = num_new_vertices // 3
            for i in range(num_new_triangles):
                v0_idx, v1_idx, v2_idx = i * 3, i * 3 + 1, i * 3 + 2
                new_tri_clip[0] = temp_poly_array[v0_idx]
                new_tri_clip[1] = temp_poly_array[v1_idx]
                new_tri_clip[2] = temp_poly_array[v2_idx]
                new_tri_uv[0] = temp_uv_array[v0_idx]
                new_tri_uv[1] = temp_uv_array[v1_idx]
                new_tri_uv[2] = temp_uv_array[v2_idx]
                calc_ndc(new_tri_clip, tri_proj_ndc)
                if calc_polygon(tri_proj_ndc, new_tri_clip, px_py, z_vals, w_vals, width, height):
                    screen_pts[tri_cnt, 0, 0] = px_py[0, 0]
                    screen_pts[tri_cnt, 0, 1] = px_py[0, 1]
                    screen_pts[tri_cnt, 1, 0] = px_py[1, 0]
                    screen_pts[tri_cnt, 1, 1] = px_py[1, 1]
                    screen_pts[tri_cnt, 2, 0] = px_py[2, 0]
                    screen_pts[tri_cnt, 2, 1] = px_py[2, 1]
                    
                    z_list[tri_cnt, 0] = z_vals[0]
                    z_list[tri_cnt, 1] = z_vals[1]
                    z_list[tri_cnt, 2] = z_vals[2]
                    
                    w_list[tri_cnt, 0] = w_vals[0]
                    w_list[tri_cnt, 1] = w_vals[1]
                    w_list[tri_cnt, 2] = w_vals[2]
                    
                    uvs_list[tri_cnt, 0, 0] = new_tri_uv[0, 0]
                    uvs_list[tri_cnt, 0, 1] = new_tri_uv[0, 1]
                    uvs_list[tri_cnt, 1, 0] = new_tri_uv[1, 0]
                    uvs_list[tri_cnt, 1, 1] = new_tri_uv[1, 1]
                    uvs_list[tri_cnt, 2, 0] = new_tri_uv[2, 0]
                    uvs_list[tri_cnt, 2, 1] = new_tri_uv[2, 1]

                    tri_indices[tri_cnt] = tri_idx
                    tri_cnt += 1

    return tri_cnt

@jit(nopython=True)
def fill_triangle(pts, color, vertices_buffer, vertex_count):
    positions = vertices_buffer['position']
    colors = vertices_buffer['color']
    texcoords = vertices_buffer['tex_coord']
    for p in pts:
        positions[vertex_count, 0] = p[0]
        positions[vertex_count, 1] = p[1]
        colors[vertex_count, 0] = color[0]
        colors[vertex_count, 1] = color[1]
        colors[vertex_count, 2] = color[2]
        colors[vertex_count, 3] = 255
        texcoords[vertex_count, 0] = 0.0
        texcoords[vertex_count, 1] = 0.0
        vertex_count += 1
    return vertex_count

@jit(nopython=True, fastmath=True)
def rasterize_triangle(pts, z_vals, w_vals, uvs, tex, framebuffer, z_buffer, object_buffer, mesh_idx, pixels_drawn, shadow, diffuse):
    h, w, _ = framebuffer.shape
    tex_h, tex_w, _ = tex.shape
    tex_w_scale = tex_w - 1
    tex_h_scale = tex_h - 1
    x0, y0 = pts[0]; x1, y1 = pts[1]; x2, y2 = pts[2]
    u0, v0 = uvs[0]; u1, v1 = uvs[1]; u2, v2 = uvs[2]
    z0, z1, z2 = z_vals[0], z_vals[1], z_vals[2]
    w0, w1, w2 = w_vals[0], w_vals[1], w_vals[2]
    invw0, invw1, invw2 = 1.0/w0, 1.0/w1, 1.0/w2
    u0 *= invw0; v0 *= invw0; u1 *= invw1; v1 *= invw1; u2 *= invw2; v2 *= invw2
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
        u0, v0, u1, v1 = u1, v1, u0, v0
        z0, z1 = z1, z0
        invw0, invw1 = invw1, invw0
    if y2 < y0:
        x0, y0, x2, y2 = x2, y2, x0, y0
        u0, v0, u2, v2 = u2, v2, u0, v0
        z0, z2 = z2, z0
        invw0, invw2 = invw2, invw0
    if y2 < y1:
        x1, y1, x2, y2 = x2, y2, x1, y1
        u1, v1, u2, v2 = u2, v2, u1, v1
        z1, z2 = z2, z1
        invw1, invw2 = invw2, invw1
    min_y = max(int(np.floor(y0)), 0)
    max_y = min(int(np.ceil(y2)), h - 1)
    area = edge_function(x0, y0, x1, y1, x2, y2)
    area_sign = np.sign(area)
    if abs(area) < 1e-12: return pixels_drawn
    for y in range(min_y, max_y + 1):
        py = y + 0.5
        x_left = get_x_on_line(x0, y0, x2, y2, py)
        if py <= y1: x_right = get_x_on_line(x0, y0, x1, y1, py)
        else: x_right = get_x_on_line(x1, y1, x2, y2, py)
        
        if x_left > x_right: x_left, x_right = x_right, x_left
        
        min_x = max(int(np.floor(x_left)), 0)
        max_x = min(int(np.ceil(x_right)), w - 1)
        for x in range(min_x, max_x + 1):
            px = x + 0.5
            w0_bary = edge_function(x1, y1, x2, y2, px, py)
            w1_bary = edge_function(x2, y2, x0, y0, px, py)
            w2_bary = edge_function(x0, y0, x1, y1, px, py)
            inside_mask = ((w0_bary * area_sign >= 0) & (w1_bary * area_sign >= 0) & (w2_bary * area_sign >= 0))
            if inside_mask:
                w0_bary /= area; w1_bary /= area; w2_bary /= area
                inv_w = w0_bary * invw0 + w1_bary * invw1 + w2_bary * invw2
                if inv_w == 0: continue
                z = w0_bary * z0 + w1_bary * z1 + w2_bary * z2
                if z < z_buffer[y, x]:
                    z_buffer[y, x] = z
                    u = (w0_bary * u0 + w1_bary * u1 + w2_bary * u2) / inv_w
                    v = (w0_bary * v0 + w1_bary * v1 + w2_bary * v2) / inv_w
                    tx = int(u * tex_w_scale)
                    ty = int((1.0 - v) * tex_h_scale)
                    if 0 <= tx < tex_w and 0 <= ty < tex_h:
                        # Get base color from texture
                        base_color = tex[ty, tx]  # Shape: (3,) with uint8 values
                    
                        # Convert to float for lighting calculation
                        r = float(base_color[0])
                        g = float(base_color[1])
                        b = float(base_color[2])
                        
                        if shadow:
                            r *= 0.3
                            g *= 0.3
                            b *= 0.3
                        else:
                            r *= diffuse
                            g *= diffuse
                            b *= diffuse
                        
                        # Clamp and convert back to uint8
                        framebuffer[y, x, 0] = np.uint8(min(r, 255.0))
                        framebuffer[y, x, 1] = np.uint8(min(g, 255.0))
                        framebuffer[y, x, 2] = np.uint8(min(b, 255.0))
                        
                        object_buffer[y, x] = mesh_idx
                        pixels_drawn += 1
    return pixels_drawn


@jit(nopython=True, parallel=True, fastmath=True)
def render_meshes_parallel(mesh_list, pivot_list, pos_list, rot_list, tex_list, uv_list, 
                        mesh_visible_flag, removed_flag, light_data_list, light_active_flag,
                        view_mat, proj_mat, width, height, 
                        vertices_buffer, framebuffer, z_buffer, object_buffer, diffuse_list,
                        thread_pts, thread_z, thread_w, thread_uv, thread_tex_idx, thread_tri_indices,
                        thread_screen_pts, thread_z_list, thread_w_list, thread_tri_indices_list, thread_uv_list,
                        mode):
    pixels_drawn = 0
    total_tris = 0
    NUM_BANDS = thread_pts.shape[0]
    for mesh in mesh_list: total_tris += len(mesh)
    pixels_drawn_per_band = np.zeros(NUM_BANDS, dtype=np.int32)
    max_tris_per_thread = ((total_tris*4 + NUM_BANDS - 1) // NUM_BANDS + 100)
    thread_counts = np.zeros(NUM_BANDS, dtype=np.int32)
    diffuse_buffer = np.ones((len(mesh_list), MAX_TRIS), dtype=np.float32)
    
    for mesh_idx in range(len(mesh_list)):
        if not mesh_visible_flag[mesh_idx] or removed_flag[mesh_idx]: continue 
        mesh = mesh_list[mesh_idx]
        uvs = uv_list[mesh_idx]
        tex = tex_list[mesh_idx]
        pivot = pivot_list[mesh_idx]
        pos = pos_list[mesh_idx]
        irx, iry, irz = rot_list[mesh_idx]
        diffuse_buffer[mesh_idx] = diffuse_list[mesh_idx]
        rx, ry, rz = rotation_matrices(irx, iry, irz)
        rot_mat = rz @ ry @ rx
        for band_index in prange(NUM_BANDS):
            band_size = (len(mesh) + NUM_BANDS - 1) // NUM_BANDS
            start_idx = band_index * band_size
            screen_pts = thread_screen_pts[band_index]
            z_list = thread_z_list[band_index]
            w_list = thread_w_list[band_index]
            uvs_list = thread_uv_list[band_index]
            tri_indices = thread_tri_indices_list[band_index]
            end_idx = min((band_index + 1) * band_size, len(mesh))

            tri_cnt = process_mesh(mesh, uvs, pivot, pos, rot_mat, view_mat, proj_mat, width, height, start_idx, end_idx, screen_pts, z_list, w_list, uvs_list, tri_indices)
            local_count = thread_counts[band_index]
            for j in range(tri_cnt):
                if local_count < max_tris_per_thread:
                    pts = screen_pts[j]
                    z_vals = z_list[j]
                    w_vals = w_list[j]
                    for k in range(3):
                        thread_pts[band_index, local_count, k, 0] = pts[k][0]
                        thread_pts[band_index, local_count, k, 1] = pts[k][1]
                    
                    thread_z[band_index, local_count, :] = z_vals
                    thread_w[band_index, local_count, :] = w_vals
                    thread_tex_idx[band_index, local_count] = mesh_idx
                    thread_uv[band_index, local_count, :, :] = uvs_list[j]
                    thread_tri_indices[band_index, local_count] = tri_indices[j]
                    local_count += 1
            
            thread_counts[band_index] = local_count
    if mode:
        for band_index in range(NUM_BANDS):
            pixels_drawn_thread = 0
            tri_cnt = thread_counts[band_index]
            for tri_idx in range(tri_cnt):
                pts = thread_pts[band_index, tri_idx]
                z_vals = thread_z[band_index, tri_idx]
                w_vals = thread_w[band_index, tri_idx]
                shadow = True
                diffuse_val = 1.0
                for light_idx in range(len(light_data_list)):
                    light_data = light_data_list[light_idx]  
                    shadow_map = light_data[0]  
                    light_view_proj = light_data[1]
                    if not light_active_flag[light_idx]: continue
                    shadow = check_shadow(pts, z_vals, w_vals, width, height, view_mat, proj_mat, light_view_proj, shadow_map)
                tri_uvs = thread_uv[band_index, tri_idx]
                mesh_idx = thread_tex_idx[band_index, tri_idx]
                tri_id = thread_tri_indices[band_index, tri_idx]
                tex = tex_list[mesh_idx]
                if 0 <= mesh_idx < MAX_MESHES and 0 <= tri_id < MAX_TRIS: diffuse_val = diffuse_buffer[mesh_idx, tri_id]
                pixels_drawn_thread = rasterize_triangle(pts, z_vals, w_vals, tri_uvs, tex, framebuffer, z_buffer, object_buffer, mesh_idx, pixels_drawn_thread, shadow, diffuse_val)
            pixels_drawn_per_band[band_index] = pixels_drawn_thread
        pixels_drawn = np.sum(pixels_drawn_per_band)
        return pixels_drawn
    else:
        vertex_count = 0
        for band_index in range(NUM_BANDS):
            tri_count = thread_counts[band_index]
            for i in range(tri_count):
                if vertex_count + 3 > len(vertices_buffer): break
                pts = thread_pts[band_index, i]
                color_id = np.random.randint(0, len(color_palette))
                color = color_palette[color_id]
                vertex_count = fill_triangle(pts, color, vertices_buffer, vertex_count)
        return vertex_count

@jit(nopython=True, fastmath=True)
def render_meshes_serial(mesh_list, pivot_list, pos_list, rot_list, tex_list, uv_list, 
                        mesh_visible_flag, removed_flag, light_data_list, light_active_flag, 
                        view_mat, proj_mat, width, height, 
                        vertices_buffer, framebuffer, z_buffer, object_buffer, diffuse_list, mode):
    vertex_count = 0
    pixels_drawn = 0
    max_tris = 0
    for m in mesh_list: max_tris += len(m)
    max_tris *= 4
    screen_pts = np.empty((max_tris, 3, 2), dtype=np.int32)
    z_list = np.empty((max_tris, 3), dtype=np.float32)
    w_list = np.empty((max_tris, 3), dtype=np.float32)
    uvs_list = np.empty((max_tris, 3, 2), dtype=np.float32)
    tri_indices = np.empty((max_tris), dtype=np.int32)
    for mesh_idx in range(len(mesh_list)):
        if not mesh_visible_flag[mesh_idx] or removed_flag[mesh_idx]: continue 
        mesh = mesh_list[mesh_idx]
        uvs = uv_list[mesh_idx]
        tex = tex_list[mesh_idx]
        pivot = pivot_list[mesh_idx]
        pos = pos_list[mesh_idx]
        diffuse_buffer = diffuse_list[mesh_idx]
        irx, iry, irz = rot_list[mesh_idx]
        rx, ry, rz = rotation_matrices(irx, iry, irz)
        rot_mat = rz @ ry @ rx
        tri_cnt = process_mesh(mesh, uvs, pivot, pos, rot_mat, view_mat, proj_mat, width, height, 0, len(mesh), screen_pts, z_list, w_list, uvs_list, tri_indices)
        for j in range(tri_cnt):
            pts = screen_pts[j]
            z_vals = z_list[j]
            w_vals = w_list[j]
            tri_uvs = uvs_list[j]
            tri_idx = tri_indices[j]
            shadow = True
            diffuse_val = 1.0
            for light_idx in range(len(light_data_list)):
                light_data = light_data_list[light_idx]  
                shadow_map = light_data[0]  
                light_view_proj = light_data[1]
                if not light_active_flag[light_idx]: continue
                shadow = check_shadow(pts, z_vals, w_vals, width, height, view_mat, proj_mat, light_view_proj, shadow_map)
            if 0 <= mesh_idx < MAX_MESHES and 0 <= tri_idx < MAX_TRIS: diffuse_val = diffuse_buffer[tri_idx]
            if not mode:
                color_id = np.random.randint(0, len(color_palette))
                color = color_palette[color_id]
                vertex_count = fill_triangle(pts, color, vertices_buffer, vertex_count)
            else: pixels_drawn = rasterize_triangle(pts, z_vals, w_vals, tri_uvs, tex, framebuffer, z_buffer, object_buffer, mesh_idx, pixels_drawn, shadow, diffuse_val)
    if not mode: return vertex_count
    return pixels_drawn

def render_meshes(mesh_list, pivot_list, pos_list, rot_list, tex_list, uv_list, mesh_visible_flag, removed_flag, light_data_list, light_active_flag, view_mat, proj_mat, width, height, vertices_buffer, framebuffer, z_buffer, object_buffer, diffuse_buffer, thread_buffers, parallel, mode):
    if parallel: 
        return render_meshes_parallel(mesh_list, pivot_list, pos_list, rot_list, tex_list, uv_list, mesh_visible_flag, removed_flag, light_data_list, light_active_flag, view_mat, proj_mat, width, height, vertices_buffer, framebuffer, z_buffer, object_buffer, diffuse_buffer, *thread_buffers, mode)
    else: 
        return render_meshes_serial(mesh_list, pivot_list, pos_list, rot_list, tex_list, uv_list, mesh_visible_flag, removed_flag, light_data_list, light_active_flag, view_mat, proj_mat, width, height, vertices_buffer, framebuffer, z_buffer, object_buffer, diffuse_buffer, mode)