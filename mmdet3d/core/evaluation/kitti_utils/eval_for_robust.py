import gc
# import io as sysio
import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class):
    CLASS_NAMES = ['car', 'truck', 'bus', 'other_vehices', 'pedestrian', 'motorcycle', 'bicycle']

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class]
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):       
        gt_name = gt_anno['name'][i]
        if (gt_name == current_cls_name):
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
        if gt_anno['name'][i] == 'dontcare':
            dc_bboxes.append(gt_anno['gt_bboxes'][i])
    for i in range(num_dt):
        if (dt_anno['name'][i] == current_cls_name):
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:              
                # calculate ih for bottom center
                ih = (
                    min(boxes[i, 2] + boxes[i, 5], qboxes[j, 2] + qboxes[j, 5])) \
                    - max(boxes[i, 2], qboxes[j, 2])

                if ih > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = ih * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):
    
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    cl_size = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        cl_size += 1
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[i, j] # revise to (gt_size, det_size)
            dt_score = dt_scores[j]
            # To get thresholds list only use this
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            # Below 2 elif make tp/fn can made
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True        
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


# @numba.jit(nopython=True) # casuse segfault when dataset is large (sample ~= 1000)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]          
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis.

    Args:
        gt_annos (dict)
        dt_annos (dict)
        metric (int): Eval type. 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 1:
            loc = np.concatenate(
                [a['location'][:, [0, 1]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 1]] for a in gt_annos_part], 0)
            rots = np.concatenate([-a['rotation_y'] for a in gt_annos_part], 0) # to C.W for rotate_iou
            gt_boxes = np.concatenate([loc, dims, rots], axis=1)      
            loc = np.concatenate(
                [a['location'][:, [0, 1]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 1]] for a in dt_annos_part], 0)
            rots = np.concatenate([-a['rotation_y'] for a in dt_annos_part], 0) # to C.W for rotate_iou
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            # TODO: If use loc, you have to add code converting gravitiy center to bottom center
            # loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            # dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            # rots = np.concatenate([-a['rotation_y'] for a in gt_annos_part], 0) # to C.W for rotate_iou
            # np.concatenate([loc, dims, rots], axis=1)
            gt_boxes = np.concatenate([a['gt_bboxes'] for a in gt_annos_part], 0)
            gt_boxes[:, -1] *= -1 # to C.W for rotate_iou
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([-a['rotation_y'] for a in dt_annos_part], 0) # to C.W for rotate_iou
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = clean_data(gt_annos[i], dt_annos[i], current_class)
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 7)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = gt_annos[i]['gt_bboxes']
        dt_datas = np.concatenate([np.concatenate([ 
            dt_annos[i]['location'], dt_annos[i]['dimensions'], dt_annos[i]['rotation_y'].reshape(-1, 1)], 1),
            dt_annos[i]['score'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               metric,
               min_overlaps,
               num_sample_pts,
               num_parts=200):
    """Kitti eval. support bev/3d eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict)
        dt_annos (dict)
        current_classes (list[int])
        metric (int): 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)
    rets = calculate_iou_partly(gt_annos, dt_annos, metric, num_parts) # revised from misodering args (dt_annos, gt_annos, ...)
    overlaps, parted_overlaps, total_gt_num, total_dt_num = rets # revise misordering
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)   
    precision = np.zeros(
        [num_class, num_minoverlap, num_sample_pts])
    recall = np.zeros(
        [num_class, num_minoverlap, num_sample_pts])
    for m, current_class in enumerate(current_classes):      
        rets = _prepare_data(gt_annos, dt_annos, current_class)
        # igonored -> mask for each classes
        (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
            dontcares, total_dc_num, total_num_valid_gt) = rets
        for k, min_overlap in enumerate(min_overlaps[:, m]):
            thresholdss = []
            for i in range(len(gt_annos)):       
                rets = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    min_overlap=min_overlap,
                    thresh=0.0)
                tp, fp, fn, similarity, thresholds = rets
                thresholdss += thresholds.tolist()
            thresholdss = np.array(thresholdss)
            thresholds = get_thresholds(thresholdss, total_num_valid_gt, num_sample_pts=num_sample_pts)
            thresholds = np.array(thresholds)
            pr = np.zeros([len(thresholds), 4]) # tp, fp, fn, similarity
            idx = 0
            for j, num_part in enumerate(split_parts):
                gt_datas_part = np.concatenate(
                    gt_datas_list[idx:idx + num_part], 0)
                dt_datas_part = np.concatenate(
                    dt_datas_list[idx:idx + num_part], 0)
                dc_datas_part = np.concatenate(
                    dontcares[idx:idx + num_part], 0)
                ignored_dets_part = np.concatenate(
                    ignored_dets[idx:idx + num_part], 0)
                ignored_gts_part = np.concatenate(
                    ignored_gts[idx:idx + num_part], 0)
                fused_compute_statistics(
                    parted_overlaps[j],
                    pr,
                    total_gt_num[idx:idx + num_part],
                    total_dt_num[idx:idx + num_part],
                    total_dc_num[idx:idx + num_part],
                    gt_datas_part,
                    dt_datas_part,
                    ignored_gts_part,
                    ignored_dets_part,
                    min_overlap=min_overlap,
                    thresholds=thresholds)
                idx += num_part
            for i in range(len(thresholds)):
                recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                precision[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
            for i in range(len(thresholds)):
                precision[m, k, i] = np.max(precision[m, k, i:], axis=-1)
                recall[m, k, i] = np.max(recall[m, k, i:], axis=-1)
                
    ret_dict = {
        'recall': recall,
        'precision': precision
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def get_mAP(prec, num_sample_pts=41):
    sums = 0
    divide_size = 4
    for i in range(0, prec.shape[-1], divide_size): # (0, num_sample_pts, divide_size)
        sums = sums + prec[..., i]
    return sums / len(range(0, prec.shape[-1], divide_size)) * 100


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            eval_types=['bev', '3d']):
    # min_overlaps: [num_minoverlap, num_class]
    N_SAMPLE_PTS = 41
    mAP_bev = None
    if 'bev' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, 1, min_overlaps, num_sample_pts=N_SAMPLE_PTS)
        mAP_bev = get_mAP(ret['precision'], N_SAMPLE_PTS)

    mAP_3d = None
    if '3d' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, 2, min_overlaps, num_sample_pts=N_SAMPLE_PTS)
        mAP_3d = get_mAP(ret['precision'], N_SAMPLE_PTS)
    return mAP_bev, mAP_3d

def kitti_eval_for_robust(gt_annos,
                          dt_annos,
                          current_classes,
                          eval_types=['bev', '3d']):
    """KITTI evaluation for Robust Dataset

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
        
    overlap_0_7 = np.array([0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5])
    overlap_0_5 = np.array([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  #(2, 7)
    class_to_name = {
        0: 'car',
        1: 'truck',
        2: 'bus',
        3: 'other_vehicles',
        4: 'pedestrian',
        5: 'motorcycle',
        6: 'bicycle'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int # ex) [3, 4, 6]
    min_overlaps = min_overlaps[:, current_classes] # select columns for current_classes
    mAPbev, mAP3d = do_eval(gt_annos, dt_annos, current_classes, min_overlaps, eval_types)
    result = ''
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP@{:.2f}:\n'.format(
                curcls_name, min_overlaps[i, j]))
            if mAPbev is not None:
                result += 'bev  AP:{:.4f}\n'.format(
                    mAPbev[j, i])
            if mAP3d is not None:
                result += '3d   AP:{:.4f}\n'.format(
                    mAP3d[j, i])

            # prepare results for logger
            prefix = f'Robust/{curcls_name}'
            postfix = f'{min_overlaps[i, j]}'
            if mAP3d is not None:
                ret_dict[f'{prefix}_3D_{postfix}'] = mAP3d[j, i]
            if mAPbev is not None:
                ret_dict[f'{prefix}_BEV_{postfix}'] = mAPbev[j, i]

    # calculate mAP over all classes if there are multiple classes
    if len(current_classes) > 1:
        for i in range(min_overlaps.shape[0]):
            overlap_list = ['0.70/0.50', '0.50/0.25']
            # prepare results for print
            result += ('\nOverall AP@{}, :\n'. format(overlap_list[i]))
            if mAPbev is not None:
                _mAPbev = mAPbev[:, i].mean(axis=0)
                result += 'bev  AP:{:.4f}\n'.format(_mAPbev)
            if mAP3d is not None:
                _mAP3d = mAP3d[:, i].mean(axis=0)
                result += '3d   AP:{:.4f}\n'.format(_mAP3d)
                
            # prepare results for logger
            if mAPbev is not None:
                ret_dict[f'Robust/Overall_BEV({overlap_list[i]})'] = _mAPbev
            if mAP3d is not None:
                ret_dict[f'Robust/Overall_3D({overlap_list[i]})'] = _mAP3d
            
    return result, ret_dict
