import os
import copy
import logging

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from open_clip import tokenize
from .precision import get_autocast
from timm.utils.model import unwrap_model
from open_clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template


def all_gather(tensor, group, return_tensor=False, args=None):
    """Perform an all-gather operation."""
    world_size = args.world_size
    tensor_list = [
        torch.empty_like(tensor) for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    if return_tensor:
        return torch.stack(tensor_list, dim=0)
    else:
        return tensor_list


def zero_shot_classifier(model, classnames, templates, args):
    # templates = templates + [lambda c: f'{c}.']
    model = unwrap_model(model)
    rank = args.rank
    world_size = args.world_size
    padding_classnames = copy.deepcopy(classnames)
    mod = len(classnames) % world_size
    if mod > 0:
        padding_classnames += padding_classnames[:world_size - mod]

    def _get_classname_emb(classname):
        texts = [template.format(classname) if isinstance(template, str) else template(
            classname) for template in templates]  # format with class
        texts = tokenize(texts).cuda(non_blocking=True)  # tokenize
        class_embeddings = model.encode_text(texts)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        return class_embedding

    with torch.no_grad():
        zeroshot_weights = []
        part_size = len(padding_classnames) // world_size
        for classname in (padding_classnames[part_size * rank:part_size * (rank + 1)]):
            class_embedding = _get_classname_emb(classname)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

        tensor_list = [
            torch.empty_like(zeroshot_weights) for _ in range(world_size)
        ]
        dist.all_gather(tensor_list, zeroshot_weights)
        zeroshot_weights = tensor_list
        zeroshot_weights = torch.cat(zeroshot_weights, dim=1)
        zeroshot_weights = zeroshot_weights[:, :len(classnames)]

    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    model = unwrap_model(model)
    total_batch_size = dataloader.batch_size * args.world_size
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        bar = tqdm(dataloader, unit_scale=total_batch_size)
        for images, target in bar:
            images = images.to(args.device)
            target = target.to(args.device)
            batch_size = images.size(0)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            bar.set_description(
                f'Acc@1 {acc1 / batch_size:.3f} Acc@5 {acc5 / batch_size:.3f}')
            top1 += acc1
            top5 += acc5
            n += batch_size
            del images, target, logits

    # sync top1, top5 and n
    data = torch.tensor([top1, top5, n]).cuda()
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    top1, top5, n = data.tolist()

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    results = {}

    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')

    model_without_ddp = unwrap_model(model)

    classifier_fname = os.getenv("EVAL_EMB", None)
    if classifier_fname is None or not os.path.exists(classifier_fname):
        logging.info(f'Building new zero-shot classifier: {classifier_fname}')
        text_classifier_name = 'text_classifier'
        classifier = None

        # if the text encoder is frozen
        enabled_saved_classifier = args.lock_text

        if enabled_saved_classifier:
            if hasattr(model_without_ddp, text_classifier_name):
                classifier = getattr(model_without_ddp, text_classifier_name)
        if classifier is None:
            classifier = zero_shot_classifier(
                model, imagenet_classnames, openai_imagenet_template, args)
        if enabled_saved_classifier:
            setattr(model_without_ddp, text_classifier_name, classifier)

        if classifier_fname is not None and args.local_rank == 0:
            torch.save(classifier.detach().T.cpu(), classifier_fname)
    else:
        logging.info(f'Apply saved zero-shot classifier, {classifier_fname}')
        classifier = torch.load(classifier_fname).T.cuda()

    logging.info('Using classifier')
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier,
                         data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier,
                         data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
