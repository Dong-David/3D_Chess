import numpy as np
import trimesh
import trimesh.transformations as tf 
from PIL import Image
from renderer.math3d import * 

def load_scene(path, nom_mesh=False):
    print(f"Loading file: {path}")
    scene = trimesh.load(path, force='scene')
    
    mesh_list = []
    tex_list = []
    uv_list = []
    node_list = []
    bounding_list = []
    
    pos_list = [] 
    rot_list = [] 

    for node_name in scene.graph.nodes:
        transform, geometry_name = scene.graph.get(node_name)
        
        if geometry_name is None: continue
        if geometry_name not in scene.geometry: continue
        
        mesh_obj = scene.geometry[geometry_name]
        print(f"  Loading mesh: {geometry_name} (from node {node_name})")

        mesh_copy = mesh_obj.copy()
        
        pos = np.zeros(3, dtype=np.float32)
        rot_euler = np.zeros(3, dtype=np.float32)

        if not nom_mesh:
            mesh_copy.apply_transform(transform)
            pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            rot_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        else:
            scale, shear, angles, trans, persp = tf.decompose_matrix(transform)
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale[0]
            scale_mat[1, 1] = scale[1]
            scale_mat[2, 2] = scale[2]
            mesh_copy.apply_transform(scale_mat)
    
            pos = np.array(trans, dtype=np.float32)
            rot_euler = np.array(angles, dtype=np.float32)

        box = mesh_copy.bounds.astype(np.float32)

        tris_data = mesh_copy.vertices[mesh_copy.faces].astype(np.float32)
        tris = np.ascontiguousarray(tris_data)
        num_triangles = len(tris)
        
        uvs = np.zeros((num_triangles, 3, 2), dtype=np.float32)
        if hasattr(mesh_copy, 'visual') and hasattr(mesh_copy.visual, 'uv') and mesh_copy.visual.uv is not None:
            uv_data = mesh_copy.visual.uv[mesh_copy.faces].astype(np.float32)
            uvs = np.ascontiguousarray(uv_data)
        
        texture = None
        if hasattr(mesh_copy, 'visual'):
            if hasattr(mesh_copy.visual, 'material'):
                if hasattr(mesh_copy.visual.material, 'image') and mesh_copy.visual.material.image is not None:
                    texture = np.asarray(mesh_copy.visual.material.image)
                elif hasattr(mesh_copy.visual.material, 'baseColorTexture') and mesh_copy.visual.material.baseColorTexture is not None:
                    texture = np.array(mesh_copy.visual.material.baseColorTexture)
            
            if texture is None and hasattr(mesh_copy.visual, 'material') and hasattr(mesh_copy.visual.material, 'main_color'):
                color = np.array(mesh_copy.visual.material.main_color)
                if len(color) == 4: color = color[:3]
                if color.dtype in [np.float32, np.float64]:
                    if np.max(color) <= 1.0: color = (color * 255).astype(np.uint8)
                    else: color = color.astype(np.uint8)
                texture = np.full((256, 256, 3), color, dtype=np.uint8)
                    
            if texture is None and type(mesh_copy.visual).__name__ == 'TextureVisuals':
                if hasattr(mesh_copy.visual, 'material') and hasattr(mesh_copy.visual.material, 'image'):
                    img = mesh_copy.visual.material.image
                    if img is not None:
                        texture = np.array(img)
        if texture is None: 
            texture = np.full((256, 256, 3), 255, dtype=np.uint8)
        else:
            if len(texture.shape) == 2: 
                texture = np.stack([texture, texture, texture], axis=2)
            elif len(texture.shape) == 3:
                if texture.shape[2] == 4:
                    texture_data = texture[:, :, :3]
                    texture = np.ascontiguousarray(texture_data)
                elif texture.shape[2] not in [3, 4]: 
                    texture = np.full((256, 256, 3), 255, dtype=np.uint8)
            else: 
                texture = np.full((256, 256, 3), 255, dtype=np.uint8)

        if texture is not None and texture.size > 0:
            texture_float = texture.astype(np.float32) / 255.0
            texture_float = np.power(texture_float, 0.7) 
            texture = (texture_float * 255).astype(np.uint8)
        
        texture = np.ascontiguousarray(texture, dtype=np.uint8)
        
        print(f"    Triangles: {num_triangles}, Texture: {texture.shape}")
        
        mesh_list.append(tris)
        tex_list.append(texture)
        uv_list.append(uvs)
        node_list.append(node_name)
        bounding_list.append(box)
        
        pos_list.append(pos)
        rot_list.append(rot_euler)

    return mesh_list, tex_list, uv_list, node_list, bounding_list, pos_list, rot_list

def load_mesh(path, nom_mesh=False):
    print(f"Loading mesh: {path}")
    mesh_obj = trimesh.load(path, force='mesh')
    
    mesh_copy = mesh_obj.copy()
    
    pos = np.zeros(3, dtype=np.float32)
    rot_euler = np.zeros(3, dtype=np.float32)

    if nom_mesh:
        all_verts = mesh_copy.vertices
        mesh_center = np.mean(all_verts, axis=0)
        mesh_copy.vertices -= mesh_center
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rot_euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
    box = mesh_copy.bounds.astype(np.float32)
    
    tris_data = mesh_copy.vertices[mesh_copy.faces].astype(np.float32)
    tris = np.ascontiguousarray(tris_data)
    num_triangles = len(tris)
    
    uvs = np.zeros((num_triangles, 3, 2), dtype=np.float32)
    if hasattr(mesh_copy, 'visual') and hasattr(mesh_copy.visual, 'uv') and mesh_copy.visual.uv is not None:
        uv_data = mesh_copy.visual.uv[mesh_copy.faces].astype(np.float32)
        uvs = np.ascontiguousarray(uv_data)
        
    texture = None
    if hasattr(mesh_copy, 'visual'):
        if hasattr(mesh_copy.visual, 'material'):
            if hasattr(mesh_copy.visual.material, 'image') and mesh_copy.visual.material.image is not None:
                texture = np.asarray(mesh_copy.visual.material.image)
            elif hasattr(mesh_copy.visual.material, 'baseColorTexture') and mesh_copy.visual.material.baseColorTexture is not None:
                texture = np.array(mesh_copy.visual.material.baseColorTexture)
        
        if texture is None and hasattr(mesh_copy.visual, 'material') and hasattr(mesh_copy.visual.material, 'main_color'):
            color = np.array(mesh_copy.visual.material.main_color)
            if len(color) == 4: color = color[:3]
            if color.dtype in [np.float32, np.float64]:
                if np.max(color) <= 1.0: color = (color * 255).astype(np.uint8)
                else: color = color.astype(np.uint8)
            texture = np.full((256, 256, 3), color, dtype=np.uint8)
                
        if texture is None and type(mesh_copy.visual).__name__ == 'TextureVisuals':
            if hasattr(mesh_copy.visual, 'material') and hasattr(mesh_copy.visual.material, 'image'):
                img = mesh_copy.visual.material.image
                if img is not None: texture = np.array(img)

    if texture is None: 
        texture = np.full((256, 256, 3), 255, dtype=np.uint8)
    else:
        if len(texture.shape) == 2: 
            texture = np.stack([texture, texture, texture], axis=2)
        elif len(texture.shape) == 3:
            if texture.shape[2] == 4:
                texture_data = texture[:, :, :3]
                texture = np.ascontiguousarray(texture_data)
            elif texture.shape[2] not in [3, 4]: 
                texture = np.full((256, 256, 3), 255, dtype=np.uint8)
        else: 
            texture = np.full((256, 256, 3), 255, dtype=np.uint8)
            
    if texture is not None and texture.size > 0:
        texture_float = texture.astype(np.float32) / 255.0
        texture_float = np.power(texture_float, 0.7)
        texture = (texture_float * 255).astype(np.uint8)

    texture = np.ascontiguousarray(texture, dtype=np.uint8)
    print(f"  Triangles: {num_triangles}, Texture: {texture.shape}")
    print(f"Loaded: {path}")
    
    return [tris], [texture], [uvs], ["root"], [box], [pos], [rot_euler]