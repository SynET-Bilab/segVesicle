#!/usr/bin/env python
import os
import fire
import json
import numpy as np
import pandas as pd

from segVesicle.bin.util.structures import VesicleList, Surface


def main(path : str = '.',
         stim : str = '',
         use2D : bool = True,
         batch : str = 'segVesicle.batch',
         check : str = 'segVesicle_QCheckBox_state.json',
         output_file : str = 'vesicle_all_statistics.xlsx'):
    '''
    @param path: str, 'stack-out' path. Default is the working dir;
    @param stim: str, stimulation condition like 'ctrl'. Default is empty.
    '''
    
    abspath = os.path.abspath(path)
    batch_file = '{}/{}'.format(abspath, batch)
    json_file = '{}/{}'.format(abspath, check)
    
    with open(json_file, 'r') as j:
        check_state = json.load(j)
    with open(batch_file, 'r') as f:
        items = f.readlines()
        
    mask = []
    for item in items:
        syn_name = item.rstrip()
        if not syn_name in check_state:
            continue
        if check_state[syn_name]:
            path_xml = os.path.join(abspath, syn_name, 'ves_seg', 'vesicle_analysis', '{}.xml'.format(syn_name.split('-')[0]))
            membrane = os.path.join(abspath, syn_name, 'ves_seg', 'membrane', '{}.mod'.format(syn_name.split('-')[0]))
            manual_mem = os.path.join(abspath, syn_name, 'ves_seg', 'membrane', 'premembrane.mod')
            if not os.path.exists(path_xml):
                print('{} is checked as True, but {} not found'.format(syn_name, path_xml))
                continue
            
            vl = VesicleList()
            vl.fromXMLFile(path_xml)
            s = Surface()
            
            if os.path.exists(manual_mem):
                s.from_model_use_imod_mesh(manual_mem)
            else:
                s.from_model_auto_segment(membrane, objNum=2)
                
            print(syn_name, vl._pixelSize, len(vl))
            stimulation = stim
            batchID = abspath.split('/')[-2]
            synID = syn_name
            preArea = 1.714**2 * s.surface_area()
            pixel_size = vl.getPixelSize()
            
            for i, sv in enumerate(vl):
                vesicleID = sv.getId()
                radius_px = sv.getRadius()
                if use2D:
                    radius_px = sv.getRadius2D()
                    try:
                        distance_px = sv.getDistance2D()
                    except:
                        distance_px = sv.getDistance()
                type_sv = sv.getType()
                
                if len(radius_px) == 2:
                    r1_px, r2_px = radius_px
                    r3_px = np.nan
                    r1, r2 = r1_px * pixel_size, r2_px * pixel_size
                    r3 = np.nan
                else:
                    r1_px, r2_px, r3_px = radius_px
                    r1, r2, r3 = r1_px * pixel_size, r2_px * pixel_size, r3_px * pixel_size
                distance = distance_px * pixel_size
                if r3 is np.nan:
                    diameter = 2 * (r1 + r2) / 2
                else:
                    diameter = 2 * (r1 + r2 + r3) / 3
                    
                sv_info = np.array([stimulation, batchID, synID, preArea, pixel_size, vesicleID,
                        r1_px, r2_px, r3_px, distance_px, type_sv,
                        r1, r2, r3, distance, diameter])
                mask.append(sv_info)
    
    mask = np.array(mask).astype(object)
    mask[:, 3:5] = mask[:, 3:5].astype(np.float64)
    mask[:, 5] = mask[:, 5].astype(int)  # vesicleID
    mask[:, 6:10] = mask[:, 6:10].astype(np.float64)  # 10 is vesicle type
    mask[:, 11:] = mask[:, 11:].astype(np.float64)
    
    columns = [
        'Stimulation', 'BatchID', 'SynID', 'PreArea/ nm^2', 'PixelSize',
        'VesicleID', 'Radius_r1_px', 'Radius_r2_px', 'Radius_r3_px', 'Distance_d', 'type_t',
        'Radius_r1_nm', 'Radius_r2_nm', 'Radius_r3_nm', 'Distance_d_nm', 'Diameter_nm'
    ]
    df = pd.DataFrame(mask, columns=columns)
    df.to_excel(os.path.join(path, output_file), index=False)


if __name__ == '__main__':
    fire.Fire(main)