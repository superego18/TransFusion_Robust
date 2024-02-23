import json
from mmdet3d.core.bbox import box_np_ops
import numpy as np

def create_robust_infos(idx):
    
    data_root = '/mnt/e/Robust/transfusion'
        
    channel_list = ['front', 'left', 'rear', 'right'] # counterclockwise from front side
    
    info = dict()
    
    image_dict = dict()
    image_dict['image_idx'] = idx
    image_dict['image_path'] = [f'{data_root}/raw/camera/{chnl}/{idx}.jpg' for chnl in channel_list]
    image_dict['image_shape'] = np.array([1200, 1920], dtype=np.int32)
    
    # point_cloud_dict
    pc_dict = dict()
    pc_dict['num_features'] = 4
    pc_dict['velodyne_path'] = f'{data_root}/raw/lidar/roof/{idx}.bin'
    
    calib_dict = dict()
    for i, chnl in enumerate(channel_list):
        
        with open(f'{data_root}/raw/calib/{chnl}/{idx}.txt', 'r') as file:
            lines = file.readlines()
        
        temp_dict=dict()
        for line in lines:
            sep = line.strip().split(':')
            temp_dict[sep[0]] = sep[1].strip().split(' ')

        # use float32
        Tr = np.vstack([np.array(temp_dict['Tr_velo_to_cam']).reshape(-1, 4), [0, 0, 0, 1]]).astype(np.float32)
        P = np.array(temp_dict['P2']).reshape(-1, 4).astype(np.float32)
        
        calib_dict[f'P{i}'] = P
        calib_dict[f'Tr_velo_to_cam{i}'] = Tr
    
    # every R0_rect is same
    calib_dict['R0_rect'] = np.array([0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4).astype(np.float32)

    anno_dict = dict()
    anno_file_path = f'{data_root}/annos/lidar/roof/{idx}.json'
    with open(anno_file_path, 'r') as file:
        anno_data = json.load(file)
        
    name_list = []
    location_list = []
    dimensions_list = []
    rotation_y_list = []

    for anno in anno_data['annotations']:
        if anno['3dbbox.category'] == 'other vehicles':
            name_list.append('other_vehicles')
        else:
            name_list.append(anno['3dbbox.category'])

        location_list.append(anno['3dbbox.location'])
        dimensions_list.append(anno['3dbbox.dimension'])
        rotation_y_list.append(anno['3dbbox.rotation_y']) # to c.c.w for z axis for box_np_ops
    
    anno_dict['name'] = np.array(name_list)
    anno_dict['location'] = np.array(location_list).reshape(-1, 3)
    anno_dict['dimensions'] = np.array(dimensions_list).reshape(-1, 3)
    anno_dict['rotation_y'] = np.array(rotation_y_list).reshape(-1, 1)
    
    # calculate num_points_in_gt
    points = np.fromfile(pc_dict['velodyne_path'], dtype=np.float32, count=-1).reshape([-1, 4])
    gt_bboxes = np.concatenate([anno_dict['location'], anno_dict['dimensions'], anno_dict['rotation_y']], axis=1).astype(np.float32)
    
    # gravity center to bottom center
    gt_bboxes[:, 2] = gt_bboxes[:, 2] - gt_bboxes[:, 5] * 0.5
    
    # bottom center: (0.5, 0.5, 0), gravity center: (0.5, 0.5, 0.5)
    anno_dict['num_points_in_gt'] = box_np_ops.points_in_rbbox(points[:, :3], gt_bboxes, origin=(0.5, 0.5, 0)).sum(0)
    
    anno_dict['gt_bboxes'] = gt_bboxes
    
    info['image'] = image_dict
    info['point_cloud'] = pc_dict
    info['calib'] = calib_dict
    info['annos'] = anno_dict
    
    del(image_dict)
    del(pc_dict)
    del(calib_dict) 
    del(anno_dict)

    return info