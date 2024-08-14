import numpy as np



def ellipsoid_point(radii,trans,rot_matrix,eps=0.03):
    
    y,z,x = np.meshgrid(np.arange(-50,50),np.arange(-50,50),np.arange(-50,50))
    eps = eps
    
    ellips = (z/radii[0])**2 + (y/radii[1])**2 + (x/radii[2])**2 < 1+eps
    ellips =  ellips.astype(np.uint8)
    
    cloud = np.array(np.where(ellips==1)).T - np.array([50,50,50])
    cloud_r = np.dot(cloud,rot_matrix)
    cloud_trans = cloud_r + trans
    out = np.round(cloud_trans)
    
    return out.astype(np.int16)