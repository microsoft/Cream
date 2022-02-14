predefine_archs = {

  'fbnet_b': {
    'genotypes' : [
      'conv3', 'ir_k3_e1', 
      'ir_k3_e6', 'ir_k5_e6', 'ir_k3_e1', 'ir_k3_e1', 
      'ir_k5_e6', 'ir_k5_e3', 'ir_k3_e6', 'ir_k5_e6',
      'ir_k5_e6', 'ir_k5_e1', 'skip'    , 'ir_k5_e3',
      'ir_k5_e6', 'ir_k3_e1', 'ir_k5_e1', 'ir_k5_e3',
      'ir_k5_e6', 'ir_k5_e1', 'ir_k5_e6', 'ir_k5_e6',
      'ir_k3_e6', 'conv1', 'avgpool'],
    'strides' : [
      2, 1,
      2, 1, 1, 1, 
      2, 1, 1, 1,
      2, 1, 1, 1,
      1, 1, 1, 1,
      2, 1, 1, 1,
      1, 1, 7],
    'out_channels' : [
      16, 16,
      24, 24, 24, 24,
      32, 32, 32, 32,
      64, 64, 64, 64,
      112, 112, 112, 112,
      184, 184, 184, 184,
      352, 1984, 1984,
    ],
    'dropout_ratio' : 0.2,
    'search_space': 'fbsb',
    },

  'fbnet_hit': {
    'genotypes' : [
      'conv3', 
      'ir_k3_e3', 'ir_k3_e3', 'ir_k3_e3_r2', 'ir_k3_e3', 
      'ir_k5_e6', 'ir_k5_e6', 'ir_k3_e3', 'ir_k3_e3',
      'ir_k7_e6', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e3',
      'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e6', 'ir_k5_e6_r2',
      'ir_k7_e6', 'ir_k5_e6', 'ir_k5_e6_r2', 'ir_k5_e6',
      'ir_k3_e3', 'conv1'],
    'strides' : [
      2,
      2, 1, 1, 1, 
      2, 1, 1, 1,
      2, 1, 1, 1,
      1, 1, 1, 1,
      2, 1, 1, 1,
      1, 1],
    'out_channels' : [
      16,
      48, 48, 48, 48,
      96, 96, 96, 96,
      184, 184, 184, 184,
      256, 256, 256, 256,
      352, 352, 352, 352,
      1024, 2048
    ],
    'dropout_ratio' : 0.2,
    'search_space': 'fbsb',
    },
    
}
