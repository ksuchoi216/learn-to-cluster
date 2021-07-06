from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, obj_from_dict
from mmcv.parallel import MMDataParallel
from gae.datasets import build_dataset, build_dataloader
from gae.online_evaluation import online_evaluate


def batch_processor(model, data, train_mode):
    assert train_mode

    A_pred, loss = model(data, return_loss=True) #retrn_loss=true => output with loss 
    log_vars = OrderedDict()
    
    _, A, _, gtmat = data
    
    print('[shape] A_pred={} A={}'.format(A_pred.shape, A.shape))
    
    acc, p, r = online_evaluate(A_pred, A)
    
    log_vars['loss'] = loss.item()
    log_vars['accuracy'] = acc
    log_vars['precision'] = p
    log_vars['recall'] = r

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(gtmat))

    return outputs


def train_gae(model, cfg, logger): #called from main.py, train_gae(gae,cfg,tracking event)
    # prepare data loaders
    
    for k, v in cfg.model['kwargs'].items(): #kwargs=dict(feature_dim=256)
        setattr(cfg.train_data, k, v) #k? v?
    dataset = build_dataset(cfg.train_data) #dict(passing info related to data e.g. path, k_at_hop ...)
    # dataset = (feat, A_, one_hop_idxs, edge_labels)
    data_loaders = [
        build_dataloader(dataset,
                         cfg.batch_size_per_gpu,
                         cfg.workers_per_gpu,
                         train=True,
                         shuffle=True)
    ] # data loader for distributed system(multi proccessors)

    # train
    if cfg.distributed:
        raise NotImplementedError
    else:
        _single_train(model, data_loaders, cfg) #calling _single_train()


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy() #dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    assert paramwise_options is None
    return obj_from_dict(optimizer_cfg, torch.optim,
                         dict(params=model.parameters())) 

    """
    obj_from_dict(info, parent=None, default_args=None)

    Args:
        info (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
      any type: Object built from the dict.
      return obj_type(**args) 
      e.g. SGD(lr=0.01, momentum=0.9, weight_decay=1e-4, ...)
    """


def _single_train(model, data_loaders, cfg): #called from train_gae
    if cfg.gpus > 1:
        raise NotImplemented
    # put model on gpus
    
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda() 
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer) #look at the func above

    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level) #work_dir from --work_dir argument
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)#present related results

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
