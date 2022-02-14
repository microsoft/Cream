""" Search cell """
import json
import lib.utils.genotypes as gt

from torchscope import scope
from lib.models.model_test import ModelTest

# config
stem_multiplier = 1
n_classes = 1000
init_channels = 48
model_type = 'imagenet'
cell_file = './genotypes.json'


#stem_multiplier = 3
#n_classes = 10
#init_channels = 36
#model_type = 'cifar'
#cell_file = './genotypes.json'

def main():
    file = open(cell_file, 'r') 
    js = file.read()
    r_dict = json.loads(js)

    file.close()
    genotypes_dict = {}
    for layer_idx, genotype in r_dict.items():
        genotypes_dict[int(layer_idx)] = gt.from_str(genotype)

    model_main = ModelTest(genotypes_dict, model_type, res_stem=False, init_channel=init_channels, \
                            stem_multiplier=stem_multiplier, n_nodes=4, num_classes=n_classes)

    if 'cifar' in model_type:
        input_x = (3, 32, 32)
    elif 'imagenet' in model_type:
        input_x = (3, 224, 224)
    else:
        raise Exception("Not support dataset!")

    scope(model_main, input_size=input_x)


if __name__ == "__main__":
    main()
