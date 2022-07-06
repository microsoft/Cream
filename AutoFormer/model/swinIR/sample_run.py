import torch

from .network_swinir import SwinIR


def main():
    super_cfg = {
        'depths': [6, 6, 6, 6, 6, 6],
        'embed_dim': 96,
        'num_heads': [6, 6, 6, 6, 6, 6],
        'mlp_ratio': 4.,
    }

    cfg = {
            'rstb_num': 4,                   # Num of RSTB layers/blocks in the network
            'stl_num': 6,                    # Num STL blocks per RSTB layer
            'embed_dim': [36, 36, 36, 36],   # Per RSTB layer
            'mlp_ratio': [2., 2., 1.5, 1.5], # Per RSTB layer
            'num_heads': [4, 4, 4, 4],       # Per RSTB layer
    }

    model = SwinIR(img_size=64,
                   window_size=8,
                   depths=super_cfg['depths'], 
                   embed_dim=super_cfg['embed_dim'], 
                   num_heads=super_cfg['num_heads'], 
                   mlp_ratio=super_cfg['mlp_ratio'],
                   upsampler='pixelshuffledirect')

    model.set_sample_config(cfg)

    img = torch.rand(32, 3, 64, 64)
    out = model(img)
    print('Output shape: ', out.shape)


if __name__ == '__main__':
    main()
