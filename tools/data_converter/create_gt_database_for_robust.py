import mmcv
import numpy as np
import pickle
from mmcv import track_iter_progress
from os import path as osp

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset


def create_groundtruth_database_for_robust(dataset_class_name,
                                        data_path,
                                        info_path=None,
                                        used_classes=None,
                                        database_save_path=None,
                                        db_info_save_path=None):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_path (str): Path of the info file.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
    """
    
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)

    file_client_args = dict(backend='disk')
    dataset_cfg.update(
        test_mode=False,
        modality=dict(
            use_lidar=True,
            use_depth=False,
            use_lidar_intensity=True,
            use_camera=False,
        ),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=file_client_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                file_client_args=file_client_args)
        ])

    dataset = build_dataset(dataset_cfg)

    # if database_save_path is None:
    #     database_save_path = osp.join(data_path, 'robust_gt_database')
    # if db_info_save_path is None:
    #     db_info_save_path = osp.join(data_path, 'robust_dbinfos_train.pkl')
    
    database_save_path = '/home/chanju/transfusion/TransFusion/data/robust_gt_database'
    db_info_save_path = '/home/chanju/transfusion/TransFusion/data/robust_dbinfos_train.pkl'
    
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    for j in track_iter_progress(list(range(len(dataset)))):
        
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)

        del(input_dict)
        
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        names = annos['gt_names']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d, origin=(0.5, 0.5, 0)) # bottom center

        for i in range(num_obj):
            
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            # with open(abs_filepath, 'w') as f:
            #     gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': abs_filepath, # change from rel_filepath 
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                }
                
                del(gt_points)

                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
                    
                del(db_info)    

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
