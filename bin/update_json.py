#!/usr/bin/env python3
'''
For updating the json result file: when some points are removed in .mod file, the corresponding points in .json file should be removed.
'''

import fire

def get_json_data(json_file):
    '''
    '''
    import numpy as np
    
    with open(json_file, "r") as t:
        data = eval(t.read()).get('vesicles')
    center = []
    for c in data:
        center.append(c.get('center'))
    for coor in center:
        coor[0], coor[1], coor[2] = coor[2], coor[1], coor[0]
    center = np.around(np.array(center),2).tolist()
    return center, data


def get_point_data(file):
    '''
    get clean data from the point file
    [123, 434, 454] for example
    '''
    with open(file, "r") as f:
        original_data = f.readlines()
    clean_data = []
    for point in original_data:
        clean_data.append([float(i) for i in point.rstrip('\n').split(' ') if i != ''])
    
    return clean_data


def compare(point_file, json_file):
    '''
    '''
    point_data = get_point_data(point_file)
    json_data, whole_data = get_json_data(json_file)
    delete_count = []
    for i in range(len(json_data)):
        if not json_data[i] in point_data:
            delete_count.append(i)
    return delete_count, whole_data, json_data


def write_new_json(point_file, json_file):
    '''
    '''
    import os
    import json
    
    if point_file.endswith('.mod'):
        cmd = "model2point {} {}".format(point_file, point_file.replace('.mod', '.point'))
        os.system(cmd)
        point_file = point_file.replace('.mod', '.point')
    elif point_file.endswith('.point'):
        point_file = point_file
        
    delete_count, whole_data, json_data = compare(point_file, json_file)
    new_data = []
    for i  in range(len(whole_data)):
        if not i in delete_count:
            new_data.append(whole_data[i])
    vesicle_info = {'vesicles':new_data}
    json_file_old = json_file + '~'
    cmd_protect = 'mv {} {}'.format(json_file, json_file_old)
    os.system(cmd_protect)
    with open(json_file, "w") as f:
        json.dump(vesicle_info, f)

    return delete_count, json_data, json_file


if __name__ == "__main__":
    fire.Fire(write_new_json)