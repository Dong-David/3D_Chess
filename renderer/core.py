import sdl2
import sdl2.ext
from sdl2 import sdlgfx
import numpy as np
import numba
from numba import types
from numba.typed import List
import ctypes
import os
import time

from .setting import *
from .rasterizer import render_meshes
from .shadow_mapping import create_shadow_map
from .lighting import create_diffuse_buffer
from .geometry import *
from .math3d import *



class Renderer:
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, title="3D Renderer"):
        self.width = width
        self.height = height
        self.title = title

        sdl2.ext.init()
        sdl2.SDL_SetHint(sdl2.SDL_HINT_RENDER_VSYNC, b"0")
        self.window = sdl2.ext.Window(title, size=(width, height))
        self.renderer = sdl2.ext.Renderer(self.window, flags=sdl2.SDL_RENDERER_ACCELERATED)
        self.texture = sdl2.SDL_CreateTexture(
            self.renderer.sdlrenderer,
            sdl2.SDL_PIXELFORMAT_RGB24,
            sdl2.SDL_TEXTUREACCESS_STREAMING,
            width, height
        )
        
        self.thread_nums = min(6, os.cpu_count() - 1)
        os.environ['NUMBA_NUM_THREADS'] = str(self.thread_nums)
        
        self.mesh_list = List.empty_list(mesh_type)
        self.tex_list = List.empty_list(tex_type)
        self.uv_list = List.empty_list(uv_type)
        self.pivot_list = List.empty_list(pivot_type)
        self.pos_list = List.empty_list(pivot_type)
        
        self.rot_list = List.empty_list(rot_type)
        self.mesh_visible_flag = List.empty_list(numba.types.boolean)
        self.removed_flag = List.empty_list(numba.types.boolean)
        self.aabb_list = List()

        self.light_shadow_list = List.empty_list(light_type)
        self.light_pos_list = List.empty_list(pivot_type)
        self.light_active_flag = List.empty_list(numba.types.boolean)
        self.light_diffuse_list = List.empty_list(diff_type)
        
        self.cam_pos = np.array([0.0, 10.0, -8.0], dtype=np.float32)
        self.cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.cam_dir = (self.cam_pos - self.cam_target) / np.linalg.norm(self.cam_pos - self.cam_target)
        
        self.proj_mat = projection_matrix(DEFAULT_FOV, width / height, DEFAULT_NEAR, DEFAULT_FAR)
        
        self.cpu_raster = True
        self.parallel = True
        self.thread_buffers = None
        
        self.fps = 0.0
        self.frame_time = 0.0
        
        self._init_buffers()
        
        print(f"✅ Renderer initialized: {width}x{height}, {self.thread_nums} threads")
    
    def _alloc_thread_buffers(self, num_bands, max_tris):
        thread_pts = np.empty((num_bands, max_tris, 3, 2), np.int32)
        thread_z = np.empty((num_bands, max_tris, 3), np.float32)
        thread_w = np.empty((num_bands, max_tris, 3), np.float32)
        thread_uv = np.empty((num_bands, max_tris, 3, 2), np.float32)
        thread_tex_idx = np.empty((num_bands, max_tris), np.int32)
        thread_tri_indices = np.empty((num_bands, max_tris), dtype=np.int32)
        
        thread_screen_pts = np.empty((num_bands, max_tris, 3, 2), dtype=np.int32)
        thread_z_list = np.empty((num_bands, max_tris, 3), dtype=np.float32)
        thread_w_list = np.empty((num_bands, max_tris, 3), dtype=np.float32)
        thread_tri_indices_list = np.empty((num_bands, max_tris), dtype=np.int32)
        thread_uv_list = np.empty((num_bands, max_tris, 3, 2), np.float32)
        
        return (thread_pts, thread_z, thread_w, thread_uv, thread_tex_idx, thread_tri_indices,
                thread_screen_pts, thread_z_list, thread_w_list, thread_tri_indices_list, thread_uv_list)

    def _init_buffers(self):
        max_tris_per_thread = (MAX_TRIS + self.thread_nums - 1) // self.thread_nums + 100
        self.thread_buffers = self._alloc_thread_buffers(self.thread_nums, max_tris_per_thread * 2)
        self.vertices_buffer = np.zeros(MAX_VERTICES, dtype=vertex_dtype)
        self.framebuffer = np.ascontiguousarray(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.object_buffer = np.ascontiguousarray(np.zeros((self.height, self.width), dtype=np.int32))
        self.z_buffer = np.ascontiguousarray(np.full((self.height, self.width), 1.0, dtype=np.float32))
        self.t_diff_buffer = np.ascontiguousarray(np.ones((MAX_MESHES, 56000*2), dtype=np.float32))
        
    
    #mesh
    def add_mesh(self, position, pivot, rotation, loaded_meshes, loaded_texs, loaded_uvs, aabb_data):
        self.mesh_list.append(loaded_meshes)
        self.tex_list.append(loaded_texs)
        self.uv_list.append(loaded_uvs)
        self.aabb_list.append(aabb_data)
        self.pivot_list.append(np.array(pivot, dtype=np.float32))
        self.pos_list.append(np.array(position, dtype=np.float32))
        self.rot_list.append(np.array(rotation, dtype=np.float32))
        self.mesh_visible_flag.append(True)
        self.removed_flag.append(False)

        return len(self.mesh_list) - 1
    
    def set_mesh_transform(self, mesh_idx, position=None, pivot=None, rotation=None):
        if 0 <= mesh_idx < len(self.mesh_list):
            if position is not None:
                self.pos_list[mesh_idx] = np.array(position, dtype=np.float32)
            if pivot is not None:
                self.pivot_list[mesh_idx] = np.array(pivot, dtype=np.float32)
            if rotation is not None:
                self.rot_list[mesh_idx] = np.array(rotation, dtype=np.float32)

    def set_mesh_geometry(self, mesh_idx, mesh=None, uvs=None, tex=None):
        if 0 <= mesh_idx < len(self.mesh_list):
            if mesh is not None:
                self.mesh_list[mesh_idx] = mesh
            if uvs is not None:
                self.uv_list[mesh_idx] = uvs
            if tex is not None:
                self.tex_list[mesh_idx] = tex

    def set_mesh_visible_flag(self, mesh_idx, enabled):
        if 0 <= mesh_idx < len(self.mesh_list):
            self.mesh_visible_flag[mesh_idx] = enabled

    def remove_mesh(self, mesh_idx):
        if 0 <= mesh_idx < len(self.mesh_list):
            self.mesh_list[mesh_idx] = np.empty(0, dtype=np.float32)
            self.tex_list[mesh_idx] = np.empty(0, dtype=np.uint8)
            self.uv_list[mesh_idx] = np.empty(0, dtype=np.float32)
            self.pivot_list[mesh_idx] = np.empty(0, dtype=np.float32)
            self.rot_list[mesh_idx] = np.empty(0, dtype=np.float32)
            self.removed_flag[mesh_idx] = True

    def active_mesh(self, mesh_idx):
        if 0 <= mesh_idx < len(self.mesh_list):
            self.removed_flag[mesh_idx] = False

    
    #light
    def add_light(self, position, target):
        light_pos = np.array(position, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        light_dir = (light_pos - target) / np.linalg.norm(light_pos - target)
        light_right = np.cross(light_dir, world_up)
        if np.linalg.norm(light_right) != 0: light_right /= np.linalg.norm(light_right)
        light_up = np.cross(light_right, light_dir)
        if np.linalg.norm(light_up) != 0: light_up /= np.linalg.norm(light_up)
        light_target_pos = light_pos + light_dir
        
        light_view_mat = look_at(light_pos, light_target_pos, light_up)
        light_proj_mat = projection_matrix(90, self.width / self.height, 0.1, 2000.0)
        light_view_proj = light_proj_mat @ light_view_mat
        
        shadow_map = np.ascontiguousarray(np.full((self.height, self.width), 1.0, dtype=np.float32))
        create_shadow_map(self.mesh_list, self.pivot_list, self.pos_list, self.rot_list, 
                         light_view_mat, light_proj_mat, self.width, self.height, shadow_map, self.parallel)
        create_diffuse_buffer(self.mesh_list, self.pivot_list, self.pos_list, self.rot_list, self.light_pos_list, self.light_diffuse_list, self.t_diff_buffer, self.parallel)
        
        self.light_shadow_list.append((shadow_map, light_view_proj))
        self.light_pos_list.append(light_pos)
        self.light_active_flag.append(True)
        return len(self.light_shadow_list) - 1

    def set_light_transform(self, light_idx, position=None, target=None):
        if 0 <= light_idx < len(self.light_shadow_list): 
            if position is not None:
                self.light_pos_list[light_idx] = np.array(position, dtype=np.float32)
            if target is not None:
                light_pos = self.light_pos_list[light_idx]
                world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                light_dir = (light_pos - target) / np.linalg.norm(light_pos - target)
                light_right = np.cross(light_dir, world_up)
                light_right /= np.linalg.norm(light_right)
                light_up = np.cross(light_right, light_dir)
                light_up /= np.linalg.norm(light_up)
                light_target_pos = light_pos + light_dir
                
                light_view_mat = look_at(light_pos, light_target_pos, light_up)
                light_proj_mat = projection_matrix(90, self.width / self.height, 0.1, 2000.0)
                light_view_proj = light_proj_mat @ light_view_mat
                
                old_shadow_map = self.light_shadow_list[light_idx][0]
                self.light_shadow_list[light_idx] = (old_shadow_map, light_view_proj)

    def set_light_active_flag(self, light_idx, enabled):
        if 0 <= light_idx < len(self.light_shadow_list):
            self.light_active_flag[light_idx] = enabled

    def update_light(self):
        for light_idx in range(len(self.light_shadow_list)):
            if not self.light_active_flag[light_idx]: continue
            light_view_proj = self.light_shadow_list[light_idx][1]
            light_proj_mat = projection_matrix(90, self.width / self.height, 0.1, 2000.0)
            light_view_mat = np.linalg.solve(light_proj_mat, light_view_proj)
            
            shadow_map = self.light_shadow_list[light_idx][0]
            create_shadow_map(self.mesh_list, self.pivot_list, self.pos_list, self.rot_list,
                             light_view_mat, light_proj_mat, self.width, self.height, shadow_map, self.parallel)
            create_diffuse_buffer(self.mesh_list, self.pivot_list, self.pos_list, self.rot_list, self.light_pos_list, self.light_diffuse_list, self.t_diff_buffer, self.parallel)
    
    #camera
    def set_camera_position(self, position):
        self.cam_pos = np.array(position, dtype=np.float32)
    
    def set_camera_target(self, target):
        self.cam_target = np.array(target, dtype=np.float32)
        self.cam_dir = (self.cam_pos - self.cam_target) / np.linalg.norm(self.cam_pos - self.cam_target)
    
    def set_camera_direction(self, dir_x, dir_y, dir_z):
        self.cam_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
        self.cam_dir /= np.linalg.norm(self.cam_dir)
    
    def move_camera(self, forward=0, right=0, up=0):
        cam_right = np.cross(self.cam_dir, self.world_up)
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, self.cam_dir)
        cam_up /= np.linalg.norm(cam_up)
        
        self.cam_pos -= self.cam_dir * forward
        self.cam_pos -= cam_right * right
        self.cam_pos += self.world_up * up
    
    def rotate_camera(self, yaw, pitch):
        pitch = max(-np.pi/2+0.01, min(np.pi/2-0.01, pitch))
        self.cam_dir = np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            -np.cos(pitch) * np.cos(yaw)
        ], dtype=np.float32)
        self.cam_dir /= np.linalg.norm(self.cam_dir)
    
    def get_view_matrix(self):
        cam_right = np.cross(self.cam_dir, self.world_up)
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, self.cam_dir)
        cam_up /= np.linalg.norm(cam_up)
        cam_target = self.cam_pos + self.cam_dir
        return look_at(self.cam_pos, cam_target, cam_up)

    #render control
    def clear_scene(self):
        self.mesh_list = List.empty_list(mesh_type)
        self.tex_list = List.empty_list(tex_type)
        self.uv_list = List.empty_list(uv_type)
        self.pivot_list = List.empty_list(pivot_type)
        self.rot_list = List.empty_list(rot_type)
        self.light_shadow_list = List.empty_list(light_type)
        self.light_pos_list = List.empty_list(pivot_type)
        self.light_active_flag = List.empty_list(numba.types.boolean)
        self.mesh_visible_flag = List.empty_list(numba.types.boolean)
        self.removed_flag = List.empty_list(numba.types.boolean)

    def render_meshes(self):
        self.renderer.clear(sdl2.ext.Color(0, 0, 0))
        self.framebuffer.fill(0)
        self.z_buffer.fill(1.0)
        frame_start = time.perf_counter()
        view_mat = self.get_view_matrix()
        res = render_meshes(
            self.mesh_list, self.pivot_list, self.pos_list, self.rot_list, self.tex_list, self.uv_list, 
            self.mesh_visible_flag, self.removed_flag, self.light_shadow_list, self.light_active_flag, 
            view_mat, self.proj_mat, self.width, self.height, 
            self.vertices_buffer, self.framebuffer, self.z_buffer, self.object_buffer, self.light_diffuse_list,
            self.thread_buffers, self.parallel, self.cpu_raster)

        if self.cpu_raster:
            if res > 0:
                pixels = ctypes.c_void_p()
                pitch_ptr = ctypes.c_int()
                
                if sdl2.SDL_LockTexture(self.texture, None, ctypes.byref(pixels), ctypes.byref(pitch_ptr)) == 0:
                    ctypes.memmove(pixels, self.framebuffer.ctypes.data, self.framebuffer.nbytes)
                    sdl2.SDL_UnlockTexture(self.texture)
                
                sdl2.SDL_RenderCopy(self.renderer.sdlrenderer, self.texture, None, None)
        else:
            if res > 0:
                vertices = (sdl2.SDL_Vertex * res)()
                ctypes.memmove(ctypes.addressof(vertices), self.vertices_buffer.ctypes.data, 
                             res * ctypes.sizeof(sdl2.SDL_Vertex))
                sdl2.SDL_RenderGeometry(self.renderer.sdlrenderer, None, vertices, res, None, 0)
        
        frame_end = time.perf_counter()
        self.frame_time = frame_end - frame_start
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0

    def render_lights(self):
        for i in range(len(self.light_shadow_list)):
            light_pos = self.light_pos_list[i]
            view_mat = self.get_view_matrix()
            sx, sy, visible = point_to_screen(light_pos, view_mat, self.proj_mat, self.width, self.height, self.z_buffer)
            active = self.light_active_flag[i]
            if visible and active == True:
                distance = np.linalg.norm(self.cam_pos - light_pos)
                REFERENCE_DISTANCE = 15.0  
                BASE_RADIUS = 10.0
                SCALE_FACTOR = REFERENCE_DISTANCE * BASE_RADIUS
                MIN_RADIUS = 2
                MAX_RADIUS = 25
                new_radius = SCALE_FACTOR / (distance + 1e-6)
                final_radius = int(np.clip(new_radius, MIN_RADIUS, MAX_RADIUS))
                sdlgfx.filledCircleRGBA(self.renderer.sdlrenderer, sx, sy, final_radius, 255, 255, 0, 255)

    def render_bounding_box(self, mesh_idx, line_width, color):
        if mesh_idx < 0 or mesh_idx >= len(self.aabb_list):
            return
        r, g, b, a = color
        box = self.aabb_list[mesh_idx]  
        if box is None or box.size == 0:
            return
            
        pivot = self.pivot_list[mesh_idx]
        rot = self.rot_list[mesh_idx]
        pos = self.pos_list[mesh_idx]

        min_x, min_y, min_z = box[0]
        max_x, max_y, max_z = box[1]
        
        local_corners = np.array([
            [min_x, min_y, min_z], [max_x, min_y, min_z],
            [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z],
            [max_x, max_y, max_z], [min_x, max_y, max_z]
        ], dtype=np.float32)

        rx, ry, rz = rotation_matrices(rot[0], rot[1], rot[2])
        rot_mat = rz @ ry @ rx
        
        pivot_mat = np.eye(4, dtype=np.float32)
        pivot_mat[:3, 3] = -pivot
        unpivot_mat = np.eye(4, dtype=np.float32)
        unpivot_mat[:3, 3] = pivot
        translation_mat = np.eye(4, dtype=np.float32)
        translation_mat[:3, 3] = pos
        model_mat = translation_mat @ unpivot_mat @ rot_mat @ pivot_mat
        
        world_corners = []
        for corner in local_corners:
            corner_h = np.array([corner[0], corner[1], corner[2], 1.0], dtype=np.float32)
            world_h = model_mat @ corner_h
            world_corners.append(world_h[:3])
        
        world_corners = np.array(world_corners, dtype=np.float32)

        view_mat = self.get_view_matrix()
        proj_mat = self.proj_mat
        screen_points = []
        
        for i in range(8):
            sx, sy, visible = point_to_screen(world_corners[i], view_mat, proj_mat, self.width, self.height, self.z_buffer)
            screen_points.append((int(sx), int(sy), visible))

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), 
            (4, 5), (5, 6), (6, 7), (7, 4),  
            (0, 4), (1, 5), (2, 6), (3, 7)   
        ]

        for p1_idx, p2_idx in edges:
            sx1, sy1, visible1 = screen_points[p1_idx]
            sx2, sy2, visible2 = screen_points[p2_idx]
            
            if visible1 and visible2:
                sdlgfx.thickLineRGBA(self.renderer.sdlrenderer, sx1, sy1, sx2, sy2, line_width, r, g, b, a)

    def present(self):
        self.renderer.present()
    
    def set_window_title(self, title):
        self.window.title = title.encode('utf-8')
    
    def update_fps_display(self):
        self.set_window_title(f"{self.title} - FPS: {self.fps:.2f}")
    
    def set_render_mode(self, cpu_raster=True, parallel=True):
        self.cpu_raster = cpu_raster
        self.parallel = parallel
    
    def show(self):
        self.window.show()
    
    def cleanup(self):
        sdl2.ext.quit()
        print("✅ Renderer cleaned up")
    
    def get_stats(self):
        return {
            'fps': self.fps,
            'frame_time_ms': self.frame_time * 1000,
            'mesh_count': len(self.mesh_list),
            'light_count': len(self.light_shadow_list),
            'mode': 'CPU' if self.cpu_raster else 'GPU',
            'parallel': self.parallel
        }

