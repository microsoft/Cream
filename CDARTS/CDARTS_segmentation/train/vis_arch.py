from __future__ import division
import os
import shutil
import sys
import time
import glob
import json
import logging
import argparse
import _init_paths
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat

parser = argparse.ArgumentParser(description='parameters for sampling')
parser.add_argument('--arch_loc', default='./jsons', type=str, help='resumed model')
parser.add_argument('--save_dir', default='./archs', type=str, help='saved dict')
parser.add_argument("--Fch", default=12, type=int, help='Fch')
parser.add_argument('--stem_head_width', type=float, default=0.6666666666666666, help='base learning rate')
args = parser.parse_args()

def main():
    width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.,]
    json_files = glob.glob(os.path.join(args.arch_loc, "*.json"))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            model_dict = json.loads(f.read())

        last = model_dict["lasts"]
        save_dir = os.path.join(args.save_dir, os.path.basename(json_file).strip('.json'))
        os.makedirs(save_dir, exist_ok=True)

        try:
            for b in range(len(last)):
                if len(width_mult_list) > 1:
                    plot_op(model_dict["ops"][b], model_dict["paths"][b], width=model_dict["widths"][b], head_width=args.stem_head_width, F_base=args.Fch).savefig(os.path.join(save_dir, "ops_%d_%d.png"%(0,b)), bbox_inches="tight")
                else:
                    plot_op(model_dict["ops"][b], model_dict["paths"][b], F_base=args.Fch).savefig(os.path.join(save_dir, "ops_%d_%d.png"%(0,b)), bbox_inches="tight")
            plot_path_width(model_dict["lasts"], model_dict["paths"], model_dict["widths"]).savefig(os.path.join(save_dir, "path_width%d.png"%0))
        except:
            print("Arch: {} is invalid".format(json_file))
            shutil.rmtree(save_dir) 
            

if __name__ == '__main__':
    main() 