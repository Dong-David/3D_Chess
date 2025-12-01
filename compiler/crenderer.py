from renderer.rasterizer import *
from renderer.math3d import *
from renderer.geometry import *
from renderer.shadow_mapping import *
from renderer.lighting import *
from renderer.setting import *
from renderer.core import Renderer
import traceback

def compile_renderer():
    print("🔧 Starting selective JIT compilation diagnostics...")
    
    
    print("  Creating temporary renderer for compilation...", end=" ")
    try:
        temp_renderer = Renderer(width=800, height=600, title="Compile Test")
        print("✅")
    except Exception as e:
        print(f"❌\n  Error creating renderer: {e}")
        return False
    
    width = temp_renderer.width
    height = temp_renderer.height
    
    dummy_tri = np.zeros((3, 3), dtype=np.float32)
    dummy_mat = np.eye(4, dtype=np.float32)
    dummy_eye = np.array([0.0, 0.0, -8.0], dtype=np.float32)
    dummy_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dummy_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dummy_mesh = np.zeros((2, 3, 3), dtype=np.float32)
    dummy_uvs = np.zeros((2, 3, 2), dtype=np.float32)
    dummy_tex = np.zeros((4, 4, 3), dtype=np.uint8)
    dummy_fb = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_ob = np.zeros((height, width), dtype=np.uint8)
    dummy_zb = np.full((height, width), 1.0, dtype=np.float32)
    dummy_vb = np.zeros(5, dtype=vertex_dtype)
    dummy_pivot = np.zeros(3, dtype=np.float32)
    dummy_rot = np.zeros(3, dtype=np.float32)

    mesh_type = numba.typeof(dummy_mesh)
    uv_type = numba.typeof(dummy_uvs)
    tex_type = numba.typeof(dummy_tex)
    pivot_type = numba.typeof(dummy_pivot)
    rot_type = numba.typeof(dummy_rot)
    
    thread_buffers = temp_renderer.thread_buffers
    db = temp_renderer.light_diffuse_list
    td = temp_renderer.t_diff_buffer

    dummy_mesh_list = List.empty_list(mesh_type)
    dummy_mesh_list.append(dummy_mesh)
    dummy_visible_flag = List.empty_list(numba.types.boolean)
    dummy_visible_flag.append(True)
    dummy_removed_flag = List.empty_list(numba.types.boolean)
    dummy_removed_flag.append(False)
    dummy_light_active_flag = List.empty_list(numba.types.boolean)
    dummy_light_active_flag.append(True)
    dummy_uv_list = List.empty_list(uv_type)
    dummy_uv_list.append(dummy_uvs)
    dummy_tex_list = List.empty_list(tex_type)
    dummy_tex_list.append(dummy_tex)
    dummy_pivot_list = List.empty_list(pivot_type)
    dummy_pivot_list.append(dummy_pivot)
    dummy_rot_list = List.empty_list(rot_type)
    dummy_rot_list.append(dummy_rot)
    
    dummy_shadow_map = np.full((height, width), 1.0, dtype=np.float32)
    dummy_light_view_proj = np.eye(4, dtype=np.float32)
    dummy_light_data_list = List()
    dummy_light_data_list.append((dummy_shadow_map, dummy_light_view_proj))

    proj = projection_matrix(90.0, float(width) / height, 0.1, 1000.0)
    view = look_at(dummy_eye, dummy_target, dummy_up)
    tri_h = np.ones((3, 4), np.float32)
    tri_t = np.zeros((3, 4), np.float32)
    buf_a = np.zeros((18, 6), dtype=np.float32)
    buf_b = np.zeros((18, 6), dtype=np.float32)
    v0 = np.zeros(6, dtype=np.float32)
    v1 = np.zeros(6, dtype=np.float32)
    v2 = np.zeros(6, dtype=np.float32)
    empty_poly_array = np.zeros((64, 4), dtype=np.float32)
    empty_uv_array = np.zeros((64, 2), dtype=np.float32)
    
    tests = [
        ("mat4x4_identity", lambda: mat4x4_identity()),
        ("projection_matrix", lambda: projection_matrix(90.0, 1.33, 0.1, 1000.0)),
        ("rotation_matrices", lambda: rotation_matrices(0.5, 0.5, 0.5)),
        ("transform", lambda: transform(dummy_tri, dummy_mat, tri_h, tri_t)),
        ("look_at", lambda: look_at(dummy_eye, dummy_target, dummy_up)),
        ("clip_polygon_single_plane", lambda: clip_polygon_single_plane(np.zeros((3,6),dtype=np.float32), 3, 0, 1.0, np.zeros((10,6),dtype=np.float32))),
        ("clip_polygon", lambda: clip_polygon(np.ones(6,dtype=np.float32), np.ones(6,dtype=np.float32), np.full(6,2,dtype=np.float32), buf_a, buf_b)),
        ("triangulation_polygon", lambda: triangulation_polygon(np.zeros((3,4),dtype=np.float32), np.zeros((3,2),dtype=np.float32), empty_poly_array, empty_uv_array, v0, v1, v2, buf_a, buf_b)),
        ("create_shadow_map_serial", lambda: create_shadow_map(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, view, proj, width, height, dummy_zb, 0)),
        ("create_shadow_map_parallel", lambda: create_shadow_map(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, view, proj, width, height, dummy_zb, 1)),
        ("create_diffuse_buffer_serial", lambda: create_diffuse_buffer(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, dummy_pivot_list, db, td, 0)),
        ("create_diffuse_buffer_parallel", lambda: create_diffuse_buffer(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, dummy_pivot_list, db, td, 1)),
        ("render_meshes_serial",   lambda: render_meshes(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, dummy_tex_list, dummy_uv_list, dummy_visible_flag, dummy_removed_flag, dummy_light_data_list, dummy_light_active_flag, view, proj, width, height, dummy_vb, dummy_fb, dummy_zb, dummy_ob, db, thread_buffers, False, True)),
        ("render_meshes_parallel", lambda: render_meshes(dummy_mesh_list, dummy_pivot_list, dummy_pivot_list, dummy_rot_list, dummy_tex_list, dummy_uv_list, dummy_visible_flag, dummy_removed_flag, dummy_light_data_list, dummy_light_active_flag, view, proj, width, height, dummy_vb, dummy_fb, dummy_zb, dummy_ob, db, thread_buffers, True, True)),
    ]

    for name, fn in tests:
        try:
            print(f"▶ Compiling {name} ...", end=" ")
            fn()
            print("✅ OK")
        except Exception as e:
            print(f"\n❌ FAILED in {name}: {type(e).__name__} -> {e}")
            traceback.print_exc(limit=3)
            temp_renderer.cleanup()
            return False
    
    temp_renderer.cleanup()
    print("✅ Compilation test completed.")
    return True