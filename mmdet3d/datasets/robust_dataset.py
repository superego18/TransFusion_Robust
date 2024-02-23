import mmcv
import numpy as np
import tempfile
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset

@DATASETS.register_module()
class RobustDataset(Custom3DDataset):
    r"""AI Hub's Robust Dataset for ControlWorks.

    This class serves as the API for experiments on the Robust Dataset.

    Please refer to `AI Hub <https://www.aihub.or.kr>' for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0].
    """
    CLASSES = ('car', 'truck', 'bus', 'other_vehicles', 'pedestrian', 'motorcycle', 'bicycle')
    
    def __init__(self,
                 data_root,
                 ann_file,
                 num_views=4,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        
        self.num_views = num_views
        assert self.num_views <= 4
        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        self.pcd_limit_range = pcd_limit_range

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the data information.
            
                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str | None): Prefix of image files.
                - img_file_name (list[str]): List of filename of images.
                - lidar2img (list[np.ndarray], optional): Transformations from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        pts_filename = info['point_cloud']['velodyne_path']
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename
        )
        if self.modality['use_camera']:
            image_paths = info['image']['image_path'] # list of paths (for total 4 cams)
            lidar2img_rts = []
            
            for i in range(4):
                P = info['calib'][f'P{i}']
                rect = info['calib']['R0_rect']
                Trv2c = info['calib'][f'Tr_velo_to_cam{i}']
                lidar2img = P @ rect @ Trv2c
                lidar2img_rts.append(lidar2img)
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts

        if not self.test_mode:
            annos = info['annos']
            annos = self.remove_dontcare(annos)
        
            gt_names = annos['name']
        
            # bottom center
            gt_bboxes_3d = annos['gt_bboxes'] 
            # arr = CameraInstance3DBoxes(gt_bboxes_3d).tensor.clone()
            # gt_bboxes_3d = LiDARInstance3DBoxes(arr, box_dim=arr.size(-1), with_yaw=True)
            gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=7, with_yaw=True)
            
            gt_labels_3d = []
            for cat in gt_names:
                if cat in self.CLASSES:
                    gt_labels_3d.append(self.CLASSES.index(cat))
                else:
                    gt_labels_3d.append(-1)
            gt_labels_3d = np.array(gt_labels_3d).astype(np.int64)
            
            input_dict['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_names=gt_names
                )

        return input_dict

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'dontcare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'dontcare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations
    
    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]

        gt_names = set(info['annos']['name'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids
    
    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       data_format='kitti'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str | None): Output data format. Default: 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('kitti' in data_format), \
            f'invalid data_format {data_format}'

        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]: # name ==> only 'pts_bbox'
                results_ = [out[name] for out in outputs] # results = [{'boxes_3d': , 'scores_3d': , 'labels_3d': }, {}, ...]
                pklfile_prefix_ = pklfile_prefix + name
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES, pklfile_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='kitti',
                 logger=None,
                 pklfile_prefix=None,
                 show=False,
                 out_dir=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Default: 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        assert ('kitti' in metric), \
            f'invalid metric {metric}'
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval_for_robust
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                for name, result_files_ in result_files.items(): # name: only 'pts_bbox'
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval_for_robust(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval_for_robust(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx] # gt_annos
            sample_idx = info['image']['image_idx'] # idx
            
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            if len(box_dict['box3d_lidar'][0]) > 0:
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box_lidar, score, label \
                    in zip(box_preds_lidar, scores, label_preds):
                    anno['name'].append(class_names[int(label)])
                    anno['dimensions'].append(box_lidar[3:6])
                    anno['location'].append(box_lidar[:3])
                    anno['rotation_y'].append(box_lidar[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            else:
                annos.append({
                    'name': np.array([]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']))

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted. # results_ of .format_rsults, net_ouputs of .bbox2result_kitti
                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2) # base_box3d.py # Limit the range of yaw to -pi~pi

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )
