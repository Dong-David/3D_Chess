import numpy as np
import numba
from numba import types
from numba.typed import List
from numba import jit, prange

@jit(nopython=True, fastmath=True)
def clip_polygon_single_plane(polygon_in, in_count, axis, sign, polygon_out):
    if in_count == 0:
        return 0
    out_count = 0
    eps2 = 1e-10
    for i in range(in_count):
        v1 = polygon_in[i]
        v2 = polygon_in[(i + 1) % in_count]
        dist1 = v1[3] + v1[axis] * sign
        dist2 = v2[3] + v2[axis] * sign
        is_v1_inside = dist1 >= 0.0
        is_v2_inside = dist2 >= 0.0
        if is_v1_inside and is_v2_inside:
            if out_count == 0 or np.sum((v2 - polygon_out[out_count - 1])**2) > eps2:
                polygon_out[out_count] = v2
                out_count += 1
        elif is_v1_inside and not is_v2_inside:
            if (dist1 - dist2) != 0.0:
                t = dist1 / (dist1 - dist2)
                inter = v1 + (v2 - v1) * t
                if out_count == 0 or np.sum((inter - polygon_out[out_count - 1])**2) > eps2:
                    polygon_out[out_count] = inter
                    out_count += 1
        elif not is_v1_inside and is_v2_inside:
            if (dist1 - dist2) != 0.0:
                t = dist1 / (dist1 - dist2)
                inter = v1 + (v2 - v1) * t
                if out_count == 0 or np.sum((inter - polygon_out[out_count - 1])**2) > eps2:
                    polygon_out[out_count] = inter
                    out_count += 1
                if np.sum((v2 - polygon_out[out_count - 1])**2) > eps2:
                    polygon_out[out_count] = v2
                    out_count += 1
    return out_count

@jit(nopython=True)
def clip_polygon(v0, v1, v2, buf_a, buf_b):
    buf_a[0] = v0
    buf_a[1] = v1
    buf_a[2] = v2
    count_a = 3
    for axis, sign in ((0,1.0),(0,-1.0),(1,1.0),(1,-1.0),(2,1.0),(2,-1.0)):
        count_b = clip_polygon_single_plane(buf_a, count_a, axis, sign, buf_b)
        if count_b == 0:
            return buf_b[:count_b]
        buf_a[:count_b] = buf_b[:count_b]
        count_a = count_b
    return buf_a[:count_a]

@jit(nopython=True)
def append_vertex(vertex, out_clip_array, out_uv_array, idx):
    out_clip_array[idx, 0] = vertex[0]
    out_clip_array[idx, 1] = vertex[1]
    out_clip_array[idx, 2] = vertex[2]
    out_clip_array[idx, 3] = vertex[3]
    out_uv_array[idx, 0] = vertex[4]
    out_uv_array[idx, 1] = vertex[5]

@jit(nopython=True)
def triangulation_polygon(tri_proj_clip, uvs, clipped_polygon, clipped_uvs, v0, v1, v2, buf_a, buf_b):
    v0[:4] = tri_proj_clip[0]
    v1[:4] = tri_proj_clip[1]
    v2[:4] = tri_proj_clip[2]
    v0[4:] = uvs[0]
    v1[4:] = uvs[1]
    v2[4:] = uvs[2]
    polygon = clip_polygon(v0, v1, v2, buf_a, buf_b)
    num_vertices = polygon.shape[0]
    if num_vertices < 3:
        return 0
    count = 0
    if num_vertices == 3:
        append_vertex(polygon[0], clipped_polygon, clipped_uvs, count); count += 1
        append_vertex(polygon[1], clipped_polygon, clipped_uvs, count); count += 1
        append_vertex(polygon[2], clipped_polygon, clipped_uvs, count); count += 1
    else:
        v_root = polygon[0]
        for i in range(1, num_vertices - 1):
            append_vertex(v_root, clipped_polygon, clipped_uvs, count); count += 1
            append_vertex(polygon[i], clipped_polygon, clipped_uvs, count); count += 1
            append_vertex(polygon[i + 1], clipped_polygon, clipped_uvs, count); count += 1
    return count

@jit(nopython=True, fastmath=True)
def calc_polygon(tri_proj_ndc, tri_proj_clip, px_py, z_vals, w_vals, width, height):
    for i in range(3):
        x, y, z = tri_proj_ndc[i]
        if x < -1 or x > 1 or y < -1 or y > 1 or z < -1 or z > 1: return False
        px = int((x * 0.5 + 0.5) * (width - 1))
        py = int((1.0 - (y * 0.5 + 0.5)) * (height - 1))
        px_py[i, 0], px_py[i, 1] = px, py
        z_vals[i] = z
        w_vals[i] = tri_proj_clip[i, 3]
    return True