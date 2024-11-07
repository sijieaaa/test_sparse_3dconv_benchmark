

import open3d
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from open3d.visualization import gui
# c = 0.8
# # RGB
# cmaps = [
#     [0, 0, c],
#     [0, c, 0],
#     [0, c, c],
#     [c, 0, 0],
#     [c, 0, c],
#     [c, c, 0],
#     [c, c, c],
# ]
num_classes = 256
cmap = np.random.rand(num_classes, 3)
cmap[0] = [0, 0, 0]

def viz_lidar_open3dv2(posest=None, posesT=None, width=None, height=None, return_pcd_list=False,
                       colors=None, label_names=None, labelids=None):
    '''
    posest: [n,3]. Contains 3D points (or translation vectors), no rotation.
    posesT: [n,4,4] or [n,3,4]. Contains 3x3 rotation matrix and 3x1 translation vector.
    colors: [n,3]. Range [0,1].
    '''

    pcd_list = []
    # FOR1 is the base coordinate frame at (0,0,0) for reference.
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd_list = [FOR1]

    if posest is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(posest)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if labelids is not None:
            pcd.colors = o3d.utility.Vector3dVector(cmap[labelids])
        pcd_list.append(pcd)

    if posesT is not None:
        for poseT in posesT:
            poseR = poseT[:3, :3]  
            poseR= Rotation.from_matrix(poseR).as_matrix()
            poset = poseT[:3, 3] 
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
            mesh_frame.rotate(poseR, center=(0, 0, 0))
            mesh_frame.translate(poset)
            pcd_list.append(mesh_frame)

    if return_pcd_list:
        return pcd_list
    
    o3d.visualization.draw_geometries(pcd_list)



    # app = gui.Application.instance
    # app.initialize()
    # pcd = pcd_list[1]
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    # vis.show_settings = True
    # vis.add_geometry("Points", pcd)
    # for i in range(0, len(pcd.points)):
    #     vis.add_3d_label(pcd.points[i], f'{label_names[i]}')
    # vis.reset_camera_to_default()
    # app.add_window(vis)
    # app.run()




if __name__ == '__main__':
    # ---- visualize translation only
    points = np.random.rand(1000, 3)
    colors = np.random.rand(1000, 3) 
    viz_lidar_open3dv2(posest=points, colors=colors, return_pcd_list=False)


    # ---- visualize both translation and rotation
    posesT = np.random.rand(3, 4, 4)
    viz_lidar_open3dv2(posesT=posesT, return_pcd_list=False)
