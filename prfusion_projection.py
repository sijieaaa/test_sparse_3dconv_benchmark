

import numpy as np
from PIL import Image
import os
from viz_lidar import viz_lidar_open3d
import matplotlib.pyplot as plt



def project_pc_on_image(pc, img, vel_to_cam_RT, P):
    """
    pc: [n,3]
    img: PIL.Image
    vel_to_cam_RT: [3,4], cam-fix, vel-mobile
    P: [3,4] 
    """
    # -- filter point cloud
    pc = pc[pc[:,0]>0, :]
    # -- point cloud in camera coordinate system
    pc_incamcoord = vel_to_cam_RT @ np.vstack([pc.T, np.ones([1, pc.shape[0]])]) # 3xN
    # -- project to image
    pc_incamcoord = np.vstack([pc_incamcoord, np.ones([1, pc_incamcoord.shape[1]])]) # 4xN
    uv = P @ pc_incamcoord
    uv = uv.T
    uv[:,:2] = uv[:,:2] / uv[:,-1:]
    uv = uv[uv[:,0]>0, :]
    uv = uv[uv[:,0]<img.size[0], :] # wh-uv-xy
    uv = uv[uv[:,1]>0, :]
    uv = uv[uv[:,1]<img.size[1], :]
    # -- plot
    plt.imshow(img)
    plt.scatter(uv[:,0], uv[:,1], c=uv[:,2], s=1, cmap='jet') # x,y
    plt.show()
    return uv



def read_kittiraw_calib(vel_to_cam_path, cam_to_cam_path):
    with open(vel_to_cam_path, 'r') as f:
        lines = f.readlines()
    vel_to_cam_R = lines[1].replace('R: ','').replace('\n','').split(' ')
    vel_to_cam_R = np.array([float(e) for e in vel_to_cam_R])
    vel_to_cam_R = vel_to_cam_R.reshape(3, 3)
    vel_to_cam_T = lines[2].replace('T: ','').replace('\n','').split(' ')
    vel_to_cam_T = np.array([float(e) for e in vel_to_cam_T])
    vel_to_cam_RT = np.hstack([vel_to_cam_R, vel_to_cam_T[:,None]]) # 3x4

    with open(cam_to_cam_path, 'r') as f:
        lines = f.readlines()
    P_rect_02 = lines[25].replace('P_rect_02: ','').replace('\n','').split(' ')
    P_rect_02 = np.array([float(e) for e in P_rect_02])
    P_rect_02 = P_rect_02.reshape(3, 4)

    return vel_to_cam_RT, P_rect_02


# vel_to_cam_R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02])
# vel_to_cam_T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
# P_rect_02 = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03])  


vel_to_cam_path = '2011_09_26/calib_velo_to_cam.txt'
cam_to_cam_path = '2011_09_26/calib_cam_to_cam.txt'



img_path = '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000107.png'
pc_path = '2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000107.bin'

# -- read vel_to_cam
with open(vel_to_cam_path, 'r') as f:
    lines = f.readlines()
vel_to_cam_R = lines[1].replace('R: ','').replace('\n','').split(' ')
vel_to_cam_R = np.array([float(e) for e in vel_to_cam_R])
vel_to_cam_R = vel_to_cam_R.reshape(3, 3)
vel_to_cam_T = lines[2].replace('T: ','').replace('\n','').split(' ')
vel_to_cam_T = np.array([float(e) for e in vel_to_cam_T])
vel_to_cam_RT = np.hstack([vel_to_cam_R, vel_to_cam_T[:,None]]) # 3x4

# -- read cam_to_cam
with open(cam_to_cam_path, 'r') as f:
    lines = f.readlines()
P_rect_02 = lines[25].replace('P_rect_02: ','').replace('\n','').split(' ') 
P_rect_02 = np.array([float(e) for e in P_rect_02])
P_rect_02 = P_rect_02.reshape(3, 4)

img = Image.open(img_path)
pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:,:3]


# # viz_lidar_open3d(pc)
# # -- filter point cloud
# pc = pc[pc[:,0]>0, :]
# # viz_lidar_open3d(pc)
# # -- point cloud in camera coordinate system
# pc_incamcoord = vel_to_cam_RT @ np.vstack([pc.T, np.ones([1, pc.shape[0]])])
# # viz_lidar_open3d([pc_incamcoord.T, pc])
# # -- project to image
# pc_incamcoord = np.vstack([pc_incamcoord, np.ones([1, pc_incamcoord.shape[1]])]) # 4xN
# uv = P_rect_02 @ pc_incamcoord
# uv = uv / uv[2,:]
# uv = uv.T
# uv = uv[uv[:,0]>0, :]
# uv = uv[uv[:,0]<img.size[0], :]
# uv = uv[uv[:,1]>0, :]
# uv = uv[uv[:,1]<img.size[1], :]

uv = project_pc_on_image(pc, img, vel_to_cam_RT, P_rect_02)

plt.imshow(img)
plt.plot(uv[:,0], uv[:,1], 'r.')    
plt.show()


a=1
