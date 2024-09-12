

import open3d
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

c = 0.8
# RGB
cmaps = [
    [0, 0, c],
    [0, c, 0],
    [0, c, c],
    [c, 0, 0],
    [c, 0, c],
    [c, c, 0],
    [c, c, c],
]


def viz_lidar_open3d(posest=None, posesT=None, width=None, height=None):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd_list = []
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd_list = [FOR1]

    for poseT in posesT:
        poseR = poseT[:, :3]  
        poseR= Rotation.from_matrix(poseR).as_matrix()
        poset = poseT[:, 3] 
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        mesh_frame.rotate(poseR, center=(0, 0, 0))
        mesh_frame.translate(poset)
        pcd_list.append(mesh_frame)
    for frame in pcd_list:
        vis.add_geometry(frame)


    vis.run()
    vis.destroy_window()




