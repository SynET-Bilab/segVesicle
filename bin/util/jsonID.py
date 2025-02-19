#!/usr/bin/env python3
import os
import sys
import json
import mrcfile
import logging
import numpy as np

def changeID(json_file,mrc_file):
    '''only match json ID based on mrcfile'''
    with mrcfile.open(mrc_file) as m:
        mrc_data = m.data.astype(np.int16)
    with open(json_file) as f:
        ves = json.load(f)
    vesicle_info = ves['vesicles']
    for i in range(len(vesicle_info)):
        center = vesicle_info[i]['center']
        center = np.asarray(center).astype(np.int16)
        Id = mrc_data[center[0], center[1], center[2]]
        if Id!=0:
            vesicle_info[i]["name"] = f"vesicle_{Id}"
    
    with open(json_file,'w') as f:
        json.dump(ves,f)

#changeID('./pp180_vesicle copy.json','./pp180_label_vesicle.mrc')
if __name__ == "__main__":

    import argparse
    import os
    import time
    import mrcfile
    import numpy as np

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--label', type=str, required=True, help='label mrc file')
    parser.add_argument('--jsonfile', type=str, help='output json file')

    args = parser.parse_args()

    label_path = args.label
    if not os.path.exists(label_path):
        raise ValueError(f"Label file {label_path} does not exist.")

    # 设置 jsonfile 默认值为 label 的对应 json 文件
    if args.jsonfile is None:
        args.jsonfile = label_path.replace("_label_vesicle.mrc", "_vesicle.json")

    # 如果已经存在 json 文件，并且备份文件不存在，则进行备份
    backup_jsonfile = args.jsonfile.replace(".json", "_bak.json")
    if os.path.exists(args.jsonfile) and not os.path.exists(backup_jsonfile):
        os.system(f'cp {args.jsonfile} {backup_jsonfile}')

    # 生成 json 文件
    t1 = time.time()
    changeID(args.jsonfile,args.label)
    print(f'done json generating, cost {time.time() - t1} s')