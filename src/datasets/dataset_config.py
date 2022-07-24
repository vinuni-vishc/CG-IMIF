from os.path import join

_BASE_DATA_PATH = "/home/tung-trong/research/FACIL_server/data"

dataset_config = {
    'pill_base_x':{
        'path': join(_BASE_DATA_PATH, 'Pill_Base_X'),
        'resize': (256, 256), 
        'pad': None,
        'crop': None,
        'flip': True,
        'normalize': ((0.4550, 0.5239, 0.5653), (0.2460, 0.2446, 0.2252))
    },
    'pill_base_x_true_norm':{
        'path': join(_BASE_DATA_PATH, 'Pill_Base_X'),
        'resize': (256, 256), 
        'pad': None,
        'crop': None,
        'flip': True,
        'normalize': ((0.4550, 0.5239, 0.5653), (0.2460, 0.2446, 0.2252))
    },
    'pill_base_x_multistream':{
        'path':{
            'rgb': join(_BASE_DATA_PATH, 'Pill_Base_X'),
            'edge': join(_BASE_DATA_PATH, 'Pill_Base_Edge')
        },
        'resize': (256, 256), 
        'pad': {
            'rgb': None,
            'edge': 2
        },
        'crop': None,
        'flip': True,
        'normalize': {
            'rgb': ((0.4550, 0.5239, 0.5653), (0.2460, 0.2446, 0.2252)),
            'edge': ((0.1,), (0.2752,))
        }
        # 'extend_channel': 3 # only use for depth and edge image
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    }, 
    'pill_base_x_multistream_true_norm':{
        'path':{
            'rgb': join(_BASE_DATA_PATH, 'Pill_Base_X'),
            'edge': join(_BASE_DATA_PATH, 'Pill_Base_Edge')
        },
        'resize': (256, 256), 
        'pad': {
            'rgb': None,
            'edge': 2
        },
        'crop': None,
        'flip': True,
        'normalize': {
            'rgb': ((0.4550, 0.5239, 0.5653), (0.2460, 0.2446, 0.2252)),
            'edge': ((0.1,), (0.2752,))
        }
    }
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
