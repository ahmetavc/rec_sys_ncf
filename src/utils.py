"""
    Some handy functions for pytroch model training ...
"""
import torch


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(network.parameters(),
                                    lr=params['adadelta_lr'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adagrad':
        optimizer = torch.optim.Adagrad(network.parameters(),
                                    lr=params['adagrad_lr'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adamax':
        optimizer = torch.optim.Adamax(network.parameters(), 
                                                          lr=params['adamax_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'asgd':
        optimizer = torch.optim.ASGD(network.parameters(), 
                                                          lr=params['asgd_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'])
    return optimizer