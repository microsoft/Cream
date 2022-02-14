from datasets.BaseDataset import BaseDataset


class CamVid(BaseDataset):
    @classmethod
    def get_class_colors(*args):
        return [[128, 0, 0], [128, 128, 0], [128, 128, 128], [64, 0, 128],
                [192, 128, 128], [128, 64, 128], [64, 64, 0], [64, 64, 128],
                [192, 192, 128], [0, 0, 192], [0, 128, 192]]

    @classmethod
    def get_class_names(*args):
        # class counting(gtFine)
        # 2953 2811 2934  970 1296 2949 1658 2808 2891 1654 2686 2343 1023 2832
        # 359  274  142  513 1646
        return ['Building', 'Tree', 'Sky', 'Car', 'Sign-Symbol', 'Road',
                'Pedestrian', 'Fence', 'Column-Pole', 'Side-Walk', 'Bicyclist', 'Void']
