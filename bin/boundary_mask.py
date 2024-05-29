'''This function is copied from `https://github.com/IsoNet-cryoET/IsoNet/blob/master/util/filter.py`
'''
import numpy as np
from skimage.morphology import dilation, cube


def boundary_mask(tomo, mask_boundary, binning = 2):

    out = np.zeros(tomo.shape, dtype = np.int8)
    import logging
    import os
    import sys
    if mask_boundary == None:
        out = np.ones(tomo.shape, dtype = np.int8)
        return out
    if mask_boundary[-4:] == '.mod':
        os.system('model2point {} {}.point >> /dev/null'.format(mask_boundary, mask_boundary[:-4]))
        points = np.loadtxt(mask_boundary[:-4]+'.point', dtype = np.float32)/binning
    elif mask_boundary[-6:] == '.point':
        points = np.loadtxt(mask_boundary[:-6]+'.point', dtype = np.float32)/binning
    else:
        logging.error("mask boundary file should end with .mod or .point but got {} !\n".format(mask_boundary))
        sys.exit()
    
    
    def get_polygon(points):
        if len(points) == 0:
            logging.info("No polygonal mask")
            return None
        elif len(points) <= 2:
            logging.error("In {}, {} points cannot defines a polygon of mask".format(mask_boundary, len(points)))
            sys.exit()
        else:
            logging.info("In {}, {} points defines a polygon of mask".format(mask_boundary, len(points)))
            return points[:,[1,0]]
    
    if points.ndim < 2: 
        logging.error("In {}, too few points to define a boundary".format(mask_boundary))
        sys.exit()

    z1=points[-2][-1]
    z0=points[-1][-1]

    if abs(z0 - z1) < 5:
        zmin = 0
        zmax = tomo.shape[0]
        polygon = get_polygon(points)
        logging.info("In {}, all points defines a polygon with full range in z".format(mask_boundary))

    else:
        zmin = max(min(z0,z1),0) 
        zmax = min(max(z0,z1),tomo.shape[0])
        polygon = get_polygon(points[:-2])
        logging.info("In {}, the last two points defines the z range of mask".format(mask_boundary))


    zmin = int(zmin)
    zmax = int(zmax)
    if polygon is None:
        out[zmin:zmax,:,:] = 1
    else:
        from matplotlib.path import Path
        poly_path = Path(polygon)
        y, x = np.mgrid[:tomo.shape[1],:tomo.shape[2]]
        coors = np.hstack((y.reshape(-1, 1), x.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(tomo.shape[1],tomo.shape[2])
        mask = mask.astype(np.int8)
        out[zmin:zmax,:,:] = mask[np.newaxis,:,:]
        out=out.astype(np.int8)
    selem=cube(5)
    out = dilation(out, selem)

    return out