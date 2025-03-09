#!/usr/bin/env python3
import os
import fire
import mrcfile
import multiprocessing
import numpy as np

from tifffile.tifffile import imwrite
from morph import density_fit_2d
from segVesicle.bin.util.structures import VesicleList



def normalize_scale(image : np.ndarray) -> np.ndarray:
    img = image.copy()
    max_value = img.max()
    min_value = img.min()
    img_norm = (255 * (img - min_value) / (max_value - min_value)).astype(np.uint8)
    return img_norm


def set_2D_radius(synapse, path):
    '''
    '''
    # prepare path
    mrc_file = os.path.join(path, 'ves_seg/{}_wbp_corrected.mrc'.format(synapse))
    xml_file = os.path.join(path, 'ves_seg/vesicle_analysis/{}_vesicle_class.xml'.format(synapse))
    if not os.path.exists(xml_file.replace('.xml', '.xml.bak')):
        os.system('cp {} {}'.format(xml_file, xml_file.replace('.xml', '.xml.bak')))
    img_path = os.path.join(path, 'ves_seg/vesicle_analysis/images')
    if os.path.exists(img_path):
        os.system('mv {} {}'.format(img_path, img_path + '_bak'))
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    
    # prepare data to fit and show
    with mrcfile.open(mrc_file) as mrc:
        mrc_data = mrc.data
        
    padwidth = 50
    margin = 15
    mean_value = mrc_data.mean()
    data_pad = np.pad(mrc_data, padwidth, mode='constant', constant_values=mean_value)
    
    # read vesicle xml file
    vl = VesicleList()
    vl.fromXMLFile(xml_file)
    
    for i in range(len(vl)):
        center = np.round(vl[i].getCenter()).astype(np.uint16)
        radius = vl[i].getRadius().mean()
        x_init, y_init, z_init = center
        
        z_range = range(z_init - 1, z_init + 2)
        r_ma = 0
        for z in z_range:
            center_z = np.array([z, y_init, x_init])
            center_fit, evecs_fit, radii_fit, _ = density_fit_2d(mrc_data, center_z, radius)
            if radii_fit is not None:
                r_z = 0.5 * (radii_fit[1] + radii_fit[2])
                if r_z > r_ma:
                    r_ma = r_z
                    X, Y = evecs_fit[1, 2], evecs_fit[1, 1]
                    phi = np.arctan2(Y, X) - np.pi/2
                    vl[i].setCenter2D(center_fit[[2, 1, 0]] + np.array([1, 1, 1]))  # zyx to xyz
                    vl[i].setRadius2D(np.array([radii_fit[1], radii_fit[2]]))
                    vl[i].setRotation2D(phi)
                    
        radius_new = vl[i].getRadius2D().mean()
        fit_vesicle = vl[i].sample_on_vesicle(360)
        shift = np.array([
            vl[i]._center2D[0] - radius_new - margin, 
            vl[i]._center2D[1] - radius_new - margin,
            fit_vesicle[:, 2].mean()
        ])
        fit_vesicle_shift = np.round(fit_vesicle - shift).astype(np.uint16)  # local coordinate, xyz, and z=0
        
        img = data_pad[
            np.round(vl[i]._center2D[2] + padwidth - 1).astype(np.uint16),
            np.round(vl[i]._center2D[1] + padwidth - radius_new - margin - 1).astype(np.uint16): np.round(vl[i]._center2D[1] + padwidth + radius_new + margin - 1).astype(np.uint16),
            np.round(vl[i]._center2D[0] + padwidth - radius_new - margin - 1).astype(np.uint16): np.round(vl[i]._center2D[0] + padwidth + radius_new + margin - 1).astype(np.uint16)
        ]  # xml from 1, but array from 0
        img_norm = normalize_scale(img)
        out = np.array([img_norm] * 3)
        
        out[0, np.round(radius_new + margin).astype(np.uint16), np.round(radius_new + margin).astype(np.uint16)] = 255  # center
        out[0, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 255  # ellipse, RGB:(255,0,0), in red
        out[1, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0
        out[2, fit_vesicle_shift[:, 1], fit_vesicle_shift[:, 0]] = 0
        
        imwrite(os.path.join(img_path, '{}.tif'.format(vl[i].getId())), out, photometric='rgb')
        
    vl.toXMLFile(xml_file)


def main(path: str = '.', 
         cpu: int = 1,
         batch_file : str = 'segVesicle.batch',
         xml_file_tail : str = 'vesicle_class.xml'):
    """
    @param path: father path of synapse folders
    @param cpu: number of processes to run
    @param batch_file: file with synapses to process
    @param xml_file_tail: xml file, should in pixel size 1.714
    """
    batch_file = os.path.join(os.path.abspath(path), batch_file)
    with open(batch_file) as f:
        items = f.readlines()
    synapses = [item.strip() for item in items if item.strip()]
    
    if cpu <= 1:
        for synapse in synapses:
            synapse = synapse.split('-')[0]
            check_path = os.path.join(path, synapse, 'ves_seg/vesicle_analysis/{}_{}'.format(synapse, xml_file_tail))
            if os.path.exists(check_path):
                set_2D_radius(synapse, os.path.join(path, synapse))
    else:
        tasks = []
        for synapse in synapses:
            synapse = synapse.split('-')[0]
            xml_path = os.path.join(path, synapse, 'ves_seg/vesicle_analysis/{}_{}'.format(synapse, xml_file_tail))
            if os.path.exists(xml_path):
                tasks.append((synapse, os.path.join(path, synapse)))
        with multiprocessing.Pool(processes=cpu) as pool:
            pool.starmap(set_2D_radius, tasks)



if __name__ == "__main__":
    fire.Fire(main)