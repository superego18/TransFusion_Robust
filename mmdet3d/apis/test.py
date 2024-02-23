import mmcv
import torch
import mmdet3d


def single_gpu_test(model, data_loader, show=False, show_detail=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    
    if isinstance(dataset, mmdet3d.datasets.robust_dataset.RobustDataset):
        robust = True
        if show:
            print() 
            print('This is show_results method of transfusion detector made for robust.')
            print()
    
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        if show:
            if robust:
                sample_idx = data['img_metas'][0].data[0][0]['sample_idx']
                
                for i in range(len(dataset.__dict__['data_infos'])):
                    if dataset.__dict__['data_infos'][i]['image']['image_idx'] == sample_idx:
                        gt_info = dataset.__dict__['data_infos'][i]
                        break

                classes = dataset.CLASSES

                model.module.show_results_for_robust(gt_info, classes, data, result, show_detail, out_dir)
            else:
                model.module.show_results(data, result, out_dir)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
