import os; os.environ['PYOPENGL_PLATFORM']='osmesa'
print('testing pyrender with osmesa...')
import pyrender, trimesh, numpy as np
verts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=np.float32)
faces = np.array([[0,1,2],[1,2,3]])
mesh = trimesh.Trimesh(verts, faces, process=False)
scene = pyrender.Scene(bg_color=[1,1,1,1], ambient_light=(0.5,0.5,0.5))
material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.1, alphaMode='OPAQUE',
    baseColorFactor=(0.5,0.5,0.5,1.0))
py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
scene.add(py_mesh)
camera = pyrender.PerspectiveCamera(yfov=np.radians(30))
cam_pose = np.eye(4); cam_pose[2,3]=3
scene.add(camera, pose=cam_pose)
r = pyrender.OffscreenRenderer(300, 400, point_size=1.0)
print('rendering...')
color, depth = r.render(scene, flags=pyrender.RenderFlags.NONE)
r.delete()
print('done, shape:', color.shape)
