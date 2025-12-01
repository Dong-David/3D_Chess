import numba
import numpy as np
from numba import types

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_FOV = 90
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 2000.0
MAX_VERTICES = 180000
MAX_TRIS = 56000*2
MAX_MESHES = 100
AMBIENT = 0.3
INTENSITY = 0.8
MAX_CLIPPED = 64

color_palette = np.array([
    [255, 255, 255],  # WHITE
    [255, 128, 0],    # ORANGE
    [255, 255, 0],    # YELLOW
    [0, 255, 0],      # GREEN
    [0, 128, 255],    # BLUE
    [255, 0, 0],      # RED
    [128, 0, 128],    # PURPLE
], dtype=np.uint8)

dummy_mesh = np.zeros((2, 3, 3), dtype=np.float32)
dummy_uvs = np.zeros((2, 3, 2), dtype=np.float32)
dummy_tex = np.zeros((4, 4, 3), dtype=np.uint8)
dummy_pivot = np.zeros(3, dtype=np.float32)
dummy_rot = np.zeros(3, dtype=np.float32)
dummy_diff = np.ones(MAX_TRIS, dtype=np.float32)
mesh_type = numba.typeof(dummy_mesh)
uv_type = numba.typeof(dummy_uvs)
tex_type = numba.typeof(dummy_tex)
pivot_type = numba.typeof(dummy_pivot)
rot_type = numba.typeof(dummy_rot)
diff_type = numba.typeof(dummy_diff)
shadow_map_type = types.Array(types.float32, 2, 'C')
light_view_proj_type = types.Array(types.float32, 2, 'C') 
enabled_type = types.bool_
light_type = types.Tuple((shadow_map_type, light_view_proj_type))


vertex_dtype = np.dtype([
    ("position", np.float32, 2),
    ("color",    np.uint8,   4),
    ("tex_coord",np.float32, 2)
], align=True)