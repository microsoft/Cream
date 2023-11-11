import torch
import re
from collections import defaultdict


BLOCKS_PATTERNS = [
    # blocks.<stage id>.<layer id>.
    (re.compile(r"visual.blocks\.(\d+)\.(\d+)\.(.*?)$"), 'visual.blocks.{}.{}.{}'),
    # TinyViT
    (re.compile(r"layers.(\d+)\.blocks\.(\d+)\.(.*?)$"), 'layers.{}.blocks.{}.{}'),
    # ResNet
    (re.compile(r'visual.layer(\d+).(\d+).(.*?)$'), 'visual.layer{}.{}.{}'),
]

TRANS_PATTENS = [
    (re.compile(r"resblocks\.(\d+)\.(.*?)$"), 'resblocks.{}.{}'),
]


def get_depth_state(state_dict):
    state = defaultdict(list)
    tstr = None
    for k, v in state_dict.items():
        # k is the name of the parameter `v`
        for pts in [BLOCKS_PATTERNS, TRANS_PATTENS]:
            for pt, s in pts:
                match = pt.search(k)
                if match is not None:
                    if tstr is not None:
                        assert tstr == s, (tstr, s)
                    else:
                        tstr = s
                    groups = match.groups()
                    if len(groups) == 3:
                        stage_id, block_id = map(int, groups[:2])
                        postname = groups[2]
                        new_name = tstr.format(stage_id, block_id, postname)
                    else:
                        stage_id = 0
                        block_id = int(groups[0])
                        postname = groups[1]
                        new_name = tstr.format(block_id, postname)
                    assert k.endswith(new_name)
                    prename = k[:-len(new_name)]
                    stage = state[stage_id]
                    if block_id >= len(stage):
                        stage.extend([list()] * (block_id - len(stage) + 1))

                    stage[block_id].append((v, (prename, postname)))
    assert tstr is not None
    return state, tstr


def prune_param(param, shape):
    if param.numel() == 1:
        return param
    # select the front param
    sl = [slice(0, s) for s in shape]
    param = param[sl]
    assert param.shape == shape, (param.shape, shape)
    return param


def compute_dict_params(state_dict):
    params = 0
    for v in state_dict.values():
        params += v.numel()
    return params


def weight_inherit(student_state_dict, teacher_state_dict, head_dim):
    # the function will overwrite student_state_dict
    student_depth_state, tstr = get_depth_state(student_state_dict)
    teacher_depth_state, tstr2 = get_depth_state(teacher_state_dict)
    assert tstr == tstr2
    assert len(student_depth_state) == len(teacher_depth_state)
    # remap depth
    vised = set()
    for si in sorted(student_depth_state.keys()):
        student_depth = len(student_depth_state[si])
        teacher_depth = len(teacher_depth_state[si])
        # interval_front
        encoder_type = 'interval_front'
        step = teacher_depth // max(student_depth, 1)
        idx = list(range(0, student_depth * step, step))
        print(
            f'sample_method for {encoder_type}: stage: {si} depth: {teacher_depth} -> {student_depth}, idx: {idx}')

        for i, j in enumerate(idx):
            for v, (prename, postname) in teacher_depth_state[si][j]:
                try:
                    new_name = prename + tstr.format(si, i, postname)
                except:
                    new_name = ''
                if new_name not in student_state_dict:
                    # transformer
                    assert si == 0
                    new_name = prename + tstr.format(i, postname)

                assert new_name in student_state_dict, new_name
                if '.qkv.' in new_name or '.attn.in_proj_' in new_name:
                    # qkv shape: (out_dim[q, k, v], in_dim)
                    # out - q - n_heads * head_dim
                    student_v = student_state_dict[new_name]
                    student_head = student_v.size(0) // (3 * head_dim)
                    teacher_head = v.size(0) // (3 * head_dim)
                    if new_name.endswith('.qkv.weight') or new_name.endswith('.attn.in_proj_weight'):
                        # (3 * H * head_dim, in_dim)
                        student_dim = student_v.size(1)
                        teacher_dim = v.size(1)
                        student_state_dict[new_name] = v.view(3, teacher_head, head_dim, teacher_dim)[
                            :, :student_head, :, :student_dim].reshape(3 * student_head * head_dim, student_dim)
                    else:
                        assert new_name.endswith(
                            '.qkv.bias') or new_name.endswith('.attn.in_proj_bias')
                        student_state_dict[new_name] = v.view(3, teacher_head, head_dim)[
                            :, :student_head].reshape(-1,)
                else:
                    try:
                        student_state_dict[new_name] = prune_param(
                            v, student_state_dict[new_name].shape)
                    except:
                        print(new_name, v.shape)
                        raise
                vised.add(new_name)
    other_param_names = set(student_state_dict.keys()) - vised
    print('OTHER Pruned Params:', other_param_names)
    for k in other_param_names:
        student_state_dict[k] = prune_param(
            teacher_state_dict[k], student_state_dict[k].shape)
        vised.add(k)
    assert vised == set(student_state_dict.keys()), set(
        student_state_dict.keys()) - vised
    student_num_params = compute_dict_params(student_state_dict)
    teacher_num_params = compute_dict_params(teacher_state_dict)
    print(
        f'Weight Inherit: {teacher_num_params} -> {student_num_params}, {student_num_params / teacher_num_params * 100:.2f}%')
    return student_state_dict


if __name__ == '__main__':
    def weight_inherit_for_tinyvit():
        from tiny_vit import tiny_vit_5m_224, tiny_vit_21m_224
        student_model = tiny_vit_5m_224()
        teacher_model = tiny_vit_21m_224()

        student_state_dict = student_model.state_dict()
        teacher_state_dict = teacher_model.state_dict()

        weight_inherit(student_state_dict, teacher_state_dict)

        # load inherited weights
        student_model.load_state_dict(student_state_dict)

    def weight_inherit_for_open_clip_transformer():
        from open_clip.model import Transformer
        student_model = Transformer(width=256, layers=3, heads=256 // 64)
        teacher_model = Transformer(width=512, layers=12, heads=512 // 64)

        student_state_dict = student_model.state_dict()
        teacher_state_dict = teacher_model.state_dict()

        weight_inherit(student_state_dict, teacher_state_dict, head_dim=64)

        # load inherited weights
        student_model.load_state_dict(student_state_dict)

    def weight_inherit_for_open_clip_vision():
        from open_clip.model import ImageEncoder, CLIPVisionCfg
        student_cfg = CLIPVisionCfg(layers=3, width=256)
        teacher_cfg = CLIPVisionCfg(layers=6, width=512)
        student_model = ImageEncoder(256, student_cfg, quick_gelu=False)
        teacher_model = ImageEncoder(512, teacher_cfg, quick_gelu=False)

        student_state_dict = student_model.state_dict()
        teacher_state_dict = teacher_model.state_dict()

        weight_inherit(student_state_dict, teacher_state_dict, head_dim=64)

        # load inherited weights
        student_model.load_state_dict(student_state_dict)

    def weight_inherit_for_open_clip_resnet():
        from open_clip.model import ImageEncoder, CLIPVisionCfg
        # layers to identify ResNet
        student_cfg = CLIPVisionCfg(image_size=224, layers=[
                                    1, 1, 1, 1], width=64, patch_size=None)
        teacher_cfg = CLIPVisionCfg(image_size=224, layers=[
                                    2, 2, 6, 2], width=64, patch_size=None)
        student_model = ImageEncoder(64, student_cfg, quick_gelu=False)
        teacher_model = ImageEncoder(64, teacher_cfg, quick_gelu=False)

        student_state_dict = student_model.state_dict()
        teacher_state_dict = teacher_model.state_dict()

        weight_inherit(student_state_dict, teacher_state_dict, head_dim=64)

    # weight_inherit_for_tinyvit()
    weight_inherit_for_open_clip_transformer()
    weight_inherit_for_open_clip_vision()
    weight_inherit_for_open_clip_resnet()

    print("OVER")
