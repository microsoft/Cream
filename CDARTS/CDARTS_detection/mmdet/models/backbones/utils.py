import logging
import torch
from collections import OrderedDict


def load_checkpoint(model,
                    filename,
                    strict=False,
                    logger=None):


    checkpoint = torch.load(filename)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    Args:
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    state_dict_modify = state_dict.copy()
    for name, param in state_dict.items():
        ''' for mobilenet v2
        if 'features' in name:
            name = name.replace('features.','features')
        '''
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if 'conv2' in name and 'layer4.0.conv2_d2.weight' in own_state.keys():
            d1 = name.replace('conv2', 'conv2_d1')
            d1_c = own_state[d1].size(0)
            own_state[d1].copy_(param[:d1_c,:,:,:])
            state_dict_modify[d1] = param[:d1_c,:,:,:]

            d2 = name.replace('conv2', 'conv2_d2')
            d2_c = own_state[d2].size(0)
            own_state[d2].copy_(param[d1_c:d1_c+d2_c,:,:,:])
            state_dict_modify[d2] = param[d1_c:d1_c+d2_c,:,:,:]

            d3 = name.replace('conv2', 'conv2_d3')
            own_state[d3].copy_(param[d1_c+d2_c:,:,:,:])
            state_dict_modify[d3] = param[d1_c+d2_c:,:,:,:]
        else:
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    '''
    if 'layer4.0.conv2_d2.weight' in own_state.keys():
        missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    else:
        # for mobilenetv2
        own_state_set = []
        for name in set(own_state.keys()):
            own_state_set.append(name.replace('features','features.'))
        missing_keys = set(own_state_set) - set(state_dict.keys())
    '''
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)