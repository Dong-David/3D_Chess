import numpy as np
from numba import jit, prange
from numba import types
from numba.typed import List
from .math3d import rotation_matrices
from .setting import AMBIENT, INTENSITY

@jit(nopython=True, fastmath=True, parallel=True)
def create_diffuse_buffer_parallel(mesh_list, pivot_list, pos_list, rot_list, light_pos_list, diffuse_list, temp):
    n_meshes = len(mesh_list)
    n_lights = len(light_pos_list) 
    diffuse_list.clear()
    
    for m_id in prange(n_meshes):
        idx = np.int64(m_id)
        mesh = mesh_list[idx]
        n_tris = mesh.shape[0]
        
        if n_tris == 0: continue
            
        pivot = pivot_list[idx]
        pos = pos_list[idx]
        rot = rot_list[idx]
        
        rx_m, ry_m, rz_m = rotation_matrices(rot[0], rot[1], rot[2])
        rot_mat = rz_m @ ry_m @ rx_m
        rot_3x3 = np.ascontiguousarray(rot_mat[:3, :3].T)

        for i in range(n_tris):
            v0_loc = mesh[i, 0]
            v1_loc = mesh[i, 1]
            v2_loc = mesh[i, 2]
            
            v0_w = (v0_loc - pivot) @ rot_3x3 + pivot + pos
            v1_w = (v1_loc - pivot) @ rot_3x3 + pivot + pos
            v2_w = (v2_loc - pivot) @ rot_3x3 + pivot + pos
            
            cx = (v0_w[0] + v1_w[0] + v2_w[0]) * 0.333333
            cy = (v0_w[1] + v1_w[1] + v2_w[1]) * 0.333333
            cz = (v0_w[2] + v1_w[2] + v2_w[2]) * 0.333333
            
            e1x, e1y, e1z = v1_w[0]-v0_w[0], v1_w[1]-v0_w[1], v1_w[2]-v0_w[2]
            e2x, e2y, e2z = v2_w[0]-v0_w[0], v2_w[1]-v0_w[1], v2_w[2]-v0_w[2]
            
            nx = e1y*e2z - e1z*e2y
            ny = e1z*e2x - e1x*e2z
            nz = e1x*e2y - e1y*e2x
            
            n_len = np.sqrt(nx*nx + ny*ny + nz*nz)
            if n_len > 1e-9:
                nx, ny, nz = nx/n_len, ny/n_len, nz/n_len
            else:
                nx, ny, nz = 0.0, 1.0, 0.0 
            
            diffuse_sum = 0.0
            
            for l_idx in range(n_lights):
                l_pos = light_pos_list[l_idx]
                lx, ly, lz = l_pos[0], l_pos[1], l_pos[2]
                
                dx, dy, dz = lx - cx, ly - cy, lz - cz
                
                dist_sq = dx*dx + dy*dy + dz*dz
                dist = np.sqrt(dist_sq) + 1e-9
                
                dot = nx*(dx/dist) + ny*(dy/dist) + nz*(dz/dist)
                
                if dot > 0:
                    diffuse_sum += dot
            
            val = AMBIENT + diffuse_sum * INTENSITY
            if val > 1.0: val = 1.0
            
            temp[m_id, i] = val
    for m_id in range(n_meshes):
        diffuse_list.append(temp[m_id])

@jit(nopython=True, fastmath=True)
def create_diffuse_buffer_serial(mesh_list, pivot_list, pos_list, rot_list, light_pos_list, diffuse_list, temp):
    n_meshes = len(mesh_list)
    n_lights = len(light_pos_list) 
    diffuse_list.clear()
    
    for m_id in range(n_meshes):
        idx = np.int64(m_id)
        mesh = mesh_list[idx]
        n_tris = mesh.shape[0]
        
        if n_tris == 0: continue
            
        pivot = pivot_list[idx]
        pos = pos_list[idx]
        rot = rot_list[idx]
        
        rx_m, ry_m, rz_m = rotation_matrices(rot[0], rot[1], rot[2])
        rot_mat = rz_m @ ry_m @ rx_m
        rot_3x3 = np.ascontiguousarray(rot_mat[:3, :3].T)

        for i in range(n_tris):
            v0_loc = mesh[i, 0]
            v1_loc = mesh[i, 1]
            v2_loc = mesh[i, 2]
            
            v0_w = (v0_loc - pivot) @ rot_3x3 + pivot + pos
            v1_w = (v1_loc - pivot) @ rot_3x3 + pivot + pos
            v2_w = (v2_loc - pivot) @ rot_3x3 + pivot + pos
            
            cx = (v0_w[0] + v1_w[0] + v2_w[0]) * 0.333333
            cy = (v0_w[1] + v1_w[1] + v2_w[1]) * 0.333333
            cz = (v0_w[2] + v1_w[2] + v2_w[2]) * 0.333333
            
            e1x, e1y, e1z = v1_w[0]-v0_w[0], v1_w[1]-v0_w[1], v1_w[2]-v0_w[2]
            e2x, e2y, e2z = v2_w[0]-v0_w[0], v2_w[1]-v0_w[1], v2_w[2]-v0_w[2]
            
            nx = e1y*e2z - e1z*e2y
            ny = e1z*e2x - e1x*e2z
            nz = e1x*e2y - e1y*e2x
            
            n_len = np.sqrt(nx*nx + ny*ny + nz*nz)
            if n_len > 1e-9:
                nx, ny, nz = nx/n_len, ny/n_len, nz/n_len
            else:
                nx, ny, nz = 0.0, 1.0, 0.0 
            
            diffuse_sum = 0.0
            
            for l_idx in range(n_lights):
                l_pos = light_pos_list[l_idx]
                lx, ly, lz = l_pos[0], l_pos[1], l_pos[2]
                
                dx, dy, dz = lx - cx, ly - cy, lz - cz
                
                dist_sq = dx*dx + dy*dy + dz*dz
                dist = np.sqrt(dist_sq) + 1e-9
                
                dot = nx*(dx/dist) + ny*(dy/dist) + nz*(dz/dist)
                
                if dot > 0:
                    diffuse_sum += dot
            
            val = AMBIENT + diffuse_sum * INTENSITY
            if val > 1.0: val = 1.0
            
            temp[m_id, i] = val
    for m_id in range(n_meshes):
        diffuse_list.append(temp[m_id])

def create_diffuse_buffer(mesh_list, pivot_list, pos_list, rot_list, light_pos_list, diffuse_list, temp, mode):
    if mode:
        create_diffuse_buffer_parallel(mesh_list, pivot_list, pos_list, rot_list, light_pos_list, diffuse_list, temp)
    else:
        create_diffuse_buffer_serial(mesh_list, pivot_list, pos_list, rot_list, light_pos_list, diffuse_list, temp)