from dataloaders.datasets import cityscapes, kd, coco, combine_dbs, pascal, sbd
from dataloaders.segdatasets import Cityscapes, CityscapesPanoptic, COCOPanoptic
from torch.utils.data import DataLoader
import torch.utils.data.distributed

def make_data_loader(args, **kwargs):
    root = args.data_path
    if args.dist:
        print("=> Using Distribued Sampler")
        if args.dataset == 'cityscapes':
            if args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(args, root, split='retrain')
                num_class = train_set.NUM_CLASSES
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **kwargs)

                val_set = cityscapes.CityscapesSegmentation(args, root, split='val')
                test_set = cityscapes.CityscapesSegmentation(args, root, split='test')
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, **kwargs)
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler, **kwargs)

            elif args.autodeeplab == 'train_seg':
                dataset_cfg = {
                'cityscapes': dict(
                    root=args.data_path,
                    split='train',
                    is_train=True,
                    crop_size=(args.image_height, args.image_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )}
                train_set = Cityscapes(**dataset_cfg['cityscapes'])
                num_class = train_set.num_classes
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **kwargs)

                dataset_val_cfg = {
                'cityscapes': dict(
                    root=args.data_path,
                    split='val',
                    is_train=False,
                    crop_size=(args.eval_height, args.eval_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )}
                val_set = Cityscapes(**dataset_val_cfg['cityscapes'])
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, sampler=val_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)
            
            elif args.autodeeplab == 'train_seg_panoptic':
                dataset_cfg = {
                'cityscapes_panoptic': dict(
                    root=args.data_path,
                    split='train',
                    is_train=True,
                    crop_size=(args.image_height, args.image_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                train_set = CityscapesPanoptic(**dataset_cfg['cityscapes_panoptic'])
                num_class = train_set.num_classes
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **kwargs)

                dataset_val_cfg = {
                'cityscapes_panoptic': dict(
                    root=args.data_path,
                    split='val',
                    is_train=False,
                    crop_size=(args.eval_height, args.eval_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                val_set = Cityscapes(**dataset_val_cfg['cityscapes_panoptic'])
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, sampler=val_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)
            else:
                raise Exception('autodeeplab param not set properly')

            return train_loader, train_sampler, val_loader, val_sampler, num_class

        elif args.dataset == 'coco':
            if args.autodeeplab == 'train_seg_panoptic':
                dataset_cfg = {
                'coco_panoptic': dict(
                    root=args.data_path,
                    split='train2017',
                    is_train=True,
                    min_resize_value=args.image_height,
                    max_resize_value=args.image_height,
                    resize_factor=32,
                    crop_size=(args.image_height, args.image_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=1.5,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                train_set = COCOPanoptic(**dataset_cfg['coco_panoptic'])
                num_class = train_set.num_classes
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, **kwargs)


                # train_set = coco.COCOSegmentation(args, root, split='train')
                # root=args.data_path
                # val_set = coco.COCOSegmentation(args, root, split='val')
                dataset_val_cfg = {
                'coco_panoptic': dict(
                    root=args.data_path,
                    split='val2017',
                    is_train=True,
                    min_resize_value=args.image_height,
                    max_resize_value=args.image_height,
                    resize_factor=32,
                    crop_size=(args.eval_height, args.eval_width),
                    mirror=False,
                    min_scale=1,
                    max_scale=1,
                    scale_step_size=0,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                val_set = COCOPanoptic(**dataset_val_cfg['coco_panoptic'])
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
                val_loader = DataLoader(val_set, batch_size=args.batch_size*4, shuffle=False, sampler=val_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)
                
            return train_loader, train_sampler, val_loader, val_sampler, num_class
        else:
            raise NotImplementedError

    else:
        if args.dataset == 'pascal':
            train_set = pascal.VOCSegmentation(args, root, split='train')
            val_set = pascal.VOCSegmentation(args, root, split='val')
            if args.use_sbd:
                sbd_train = sbd.SBDSegmentation(args, root, split=['train', 'val'])
                train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None

            return train_loader, train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'cityscapes':
            if args.autodeeplab == 'train_seg':
                dataset_cfg = {
                'cityscapes': dict(
                    root=args.data_path,
                    split='train',
                    is_train=True,
                    crop_size=(args.image_height, args.image_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )}
                train_set = Cityscapes(**dataset_cfg['cityscapes'])
                num_class = train_set.num_classes
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)

                dataset_val_cfg = {
                'cityscapes': dict(
                    root=args.data_path,
                    split='val',
                    is_train=False,
                    crop_size=(args.eval_height, args.eval_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )}
                val_set = Cityscapes(**dataset_val_cfg['cityscapes'])
                val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
            
            elif args.autodeeplab == 'train_seg_panoptic':
                dataset_cfg = {
                'cityscapes_panoptic': dict(
                    root=args.data_path,
                    split='train',
                    is_train=True,
                    crop_size=(args.image_height, args.image_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                train_set = CityscapesPanoptic(**dataset_cfg['cityscapes_panoptic'])
                num_class = train_set.num_classes
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)

                dataset_val_cfg = {
                'cityscapes_panoptic': dict(
                    root=args.data_path,
                    split='val',
                    is_train=False,
                    crop_size=(args.eval_height, args.eval_width),
                    mirror=True,
                    min_scale=0.5,
                    max_scale=2.0,
                    scale_step_size=0.1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    semantic_only=False,
                    ignore_stuff_in_offset=True,
                    small_instance_area=4096,
                    small_instance_weight=3
                )}
                val_set = Cityscapes(**dataset_val_cfg['cityscapes_panoptic'])
                val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size//4), shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
            else:
                raise Exception('autodeeplab param not set properly')

            return train_loader, val_loader, num_class


        elif args.dataset == 'coco':
            train_set = coco.COCOSegmentation(args, root, split='train')
            val_set = coco.COCOSegmentation(args, root, split='val')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None
            return train_loader, train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'kd':
            train_set = kd.CityscapesSegmentation(args, root, split='train')
            val_set = kd.CityscapesSegmentation(args, root, split='val')
            test_set = kd.CityscapesSegmentation(args, root, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader1 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_loader2 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader1, train_loader2, val_loader, test_loader, num_class
        else:
            raise NotImplementedError
