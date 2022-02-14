import os
import math
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import time
from tqdm import tqdm
from genotypes import PRIMITIVES
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pdb import set_trace as bp
import warnings


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
            self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        # os.mkdir(os.path.join(path, 'scripts'))
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

########################## TensorRT speed_test #################################
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    MAX_BATCH_SIZE = 1
    MAX_WORKSPACE_SIZE = 1 << 30

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    DTYPE = trt.float32

    # Model
    INPUT_NAME = 'input'
    OUTPUT_NAME = 'output'

    def allocate_buffers(engine):
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
        h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        return h_input, d_input, h_output, d_output


    def build_engine(model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = MAX_WORKSPACE_SIZE
            builder.max_batch_size = MAX_BATCH_SIZE

            with open(model_file, 'rb') as model:
                parser.parse(model.read())
                last_layer = network.get_layer(network.num_layers - 1)
                network.mark_output(last_layer.get_output(0))
                return builder.build_cuda_engine(network)


    def load_input(input_size, host_buffer):
        assert len(input_size) == 4
        b, c, h, w = input_size
        dtype = trt.nptype(DTYPE)
        img_array = np.random.randn(c, h, w).astype(dtype).ravel()
        np.copyto(host_buffer, img_array)


    def do_inference(context, h_input, d_input, h_output, d_output, iterations=None):
        # Transfer input data to the GPU.
        cuda.memcpy_htod(d_input, h_input)
        # warm-up
        for _ in range(10):
            context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        # test proper iterations
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                t_start = time.time()
                for _ in range(iterations):
                    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 3)
        # Run inference.
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
        return latency


    def compute_latency_ms_tensorrt(model, input_size, iterations=None):
        model = model.cuda()
        model.eval()
        _, c, h, w = input_size
        dummy_input = torch.randn(1, c, h, w, device='cuda')
        torch.onnx.export(model, dummy_input, "model.onnx", verbose=False, input_names=["input"], output_names=["output"])
        with build_engine("model.onnx") as engine:
            h_input, d_input, h_output, d_output = allocate_buffers(engine)
            load_input(input_size, h_input)
            with engine.create_execution_context() as context:
                latency = do_inference(context, h_input, d_input, h_output, d_output, iterations=iterations)
        # FPS = 1000 / latency (in ms)
        return latency
except:
    warnings.warn("TensorRT (or pycuda) is not installed. compute_latency_ms_tensorrt() cannot be used.")
#########################################################################

def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency

                
def plot_path(lasts, paths=[]):
    '''
    paths: list of path0~path2
    '''
    assert len(paths) > 0
    path0 = paths[0]
    path1 = paths[1] if len(paths) > 1 else []
    path2 = paths[2] if len(paths) > 2 else []

    if path0[-1] != lasts[0]: path0.append(lasts[0])
    if len(path1) != 0 and path1[-1] != lasts[1]: path1.append(lasts[1])
    if len(path2) != 0 and path2[-1] != lasts[2]: path2.append(lasts[2])
    x_len = max(len(path0), len(path1), len(path2))
    f, ax = plt.subplots(figsize=(x_len, 3))
    ax.plot(np.arange(len(path0)), 2 - np.array(path0), label='1/32', lw=2.5, color='#000000', linestyle='-')#, marker='o', markeredgecolor='r', markerfacecolor='r')
    ax.plot(np.arange(len(path1)), 2 - np.array(path1) - 0.08, lw=1.8, label='1/16', color='#313131', linestyle='--')#, marker='^', markeredgecolor='b', markerfacecolor='b')
    ax.plot(np.arange(len(path2)), 2 - np.array(path2) - 0.16, lw=1.2, label='1/8', color='#5a5858', linestyle='-.')#, marker='s', markeredgecolor='m', markerfacecolor='m')
    plt.xticks(np.arange(x_len), list(range(1, x_len+1)))
    plt.yticks(np.array([0, 1, 2]), ["1/32", "1/16", "1/8"])
    plt.ylabel("Scale", fontsize=17)
    plt.xlabel("Layer", fontsize=17)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    f.tight_layout()
    plt.legend(prop={'size': 14}, loc=3)
    return f


def plot_path_width(lasts, paths=[], widths=[]):
    '''
    paths: list of path0~path2
    '''
    assert len(paths) > 0 and len(widths) > 0
    path0 = paths[0]
    path1 = paths[1] if len(paths) > 1 else []
    path2 = paths[2] if len(paths) > 2 else []
    width0 = widths[0]
    width1 = widths[1] if len(widths) > 1 else []
    width2 = widths[2] if len(widths) > 2 else []

    # just for visualization purpose
    if path0[-1] != lasts[0]: path0.append(lasts[0])
    if len(path1) != 0 and path1[-1] != lasts[1]: path1.append(lasts[1])
    if len(path2) != 0 and path2[-1] != lasts[2]: path2.append(lasts[2])
    line_updown = -0.07
    annotation_updown = 0.05; annotation_down_scale = 1.7
    x_len = max(len(path0), len(path1), len(path2))
    f, ax = plt.subplots(figsize=(x_len, 3))
    
    assert len(path0) == len(width0) + 1 or len(path0) + len(width0) == 0, "path0 %d, width0 %d"%(len(path0), len(width0))
    assert len(path1) == len(width1) + 1 or len(path1) + len(width1) == 0, "path1 %d, width1 %d"%(len(path1), len(width1))
    assert len(path2) == len(width2) + 1 or len(path2) + len(width2) == 0, "path2 %d, width2 %d"%(len(path2), len(width2))
    
    ax.plot(np.arange(len(path0)), 2 - np.array(path0), label='1/32', lw=2.5, color='#000000', linestyle='-')
    ax.plot(np.arange(len(path1)), 2 - np.array(path1) + line_updown, lw=1.8, label='1/16', color='#313131', linestyle='--')
    ax.plot(np.arange(len(path2)), 2 - np.array(path2) + line_updown*2, lw=1.2, label='1/8', color='#5a5858', linestyle='-.')
    
    annotations = {} # (idx, scale, width, down): ((x, y), width)
    for idx, width in enumerate(width2):
        annotations[(idx, path2[idx], width, path2[idx+1]-path2[idx])] = ((0.35 + idx, 2 - path2[idx] + line_updown*2 + annotation_updown - (path2[idx+1]-path2[idx])/annotation_down_scale), width)
    for idx, width in enumerate(width1):
        annotations[(idx, path1[idx], width, path1[idx+1]-path1[idx])] = ((0.35 + idx, 2 - path1[idx] + line_updown + annotation_updown - (path1[idx+1]-path1[idx])/annotation_down_scale), width)
    for idx, width in enumerate(width0):
        annotations[(idx, path0[idx], width, path0[idx+1]-path0[idx])] = ((0.35 + idx, 2 - path0[idx] + annotation_updown - (path0[idx+1]-path0[idx])/annotation_down_scale), width)
    for k, v in annotations.items():
        plt.annotate("%.2f"%v[1], v[0], fontsize=12, color='red')
    
    plt.xticks(np.arange(x_len), list(range(1, x_len+1)))
    plt.yticks(np.array([0, 1, 2]), ["1/32", "1/16", "1/8"])
    plt.ylim([-0.4, 2.5])
    plt.ylabel("Scale", fontsize=17)
    plt.xlabel("Layer", fontsize=17)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    f.tight_layout()
    plt.legend(prop={'size': 14}, loc=3)
    return f

def plot_op(ops, path, width=[], head_width=None, F_base=16):
    assert len(width) == 0 or len(width) == len(ops) - 1
    table_vals = []
    scales = {0: "1/8", 1: "1/16", 2: "1/32"}; base_scale = 3
    for idx, op in enumerate(ops):
        scale = path[idx]
        if len(width) > 0:
            if idx < len(width):
                ch = int(F_base*2**(scale+base_scale)*width[idx])
            else:
                ch = int(F_base*2**(scale+base_scale)*head_width)
        else:
            ch = F_base*2**(scale+base_scale)
        row = [idx+1, PRIMITIVES[op], scales[scale], ch]
        table_vals.append(row)

    # Based on http://stackoverflow.com/a/8531491/190597 (Andrey Sobolev)
    col_labels = ['Stage', 'Operator', 'Scale', '#Channel_out']
    plt.tight_layout()
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table = plt.table(cellText=table_vals,
                          colWidths=[0.22, 0.6, 0.25, 0.5],
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(2, 2)

    return fig

def objective_acc_lat(acc, lat, lat_target=8.3, alpha=-0.07, beta=-0.07):
    if lat <= lat_target:
        w = alpha
    else:
        w = beta
    return acc * math.pow(lat / lat_target, w)