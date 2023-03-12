import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import timm
from models import tiny_vit


class ModelsTestCase(unittest.TestCase):
    """Test for models.py"""
    def setUp(self):
        self.ckpt_names = [
            ('tiny_vit_5m_224', ['22k_distill', '22kto1k_distill', '1k']),
            ('tiny_vit_11m_224', ['22k_distill', '22kto1k_distill', '1k']),
            ('tiny_vit_21m_224', ['22k_distill', '22kto1k_distill', '1k']),
            ('tiny_vit_21m_384', ['22kto1k_distill']),
            ('tiny_vit_21m_512', ['22kto1k_distill']),
        ]

    def test_load_model(self):
        """Test for load_model"""
        for variant, pretrained_types in self.ckpt_names:
            # empty load
            with self.subTest(variant=variant, pretrained_type='empty'):
                model = timm.create_model(variant)
                assert model.head.weight.shape[0] == 1000
            # load pretrained
            for pretrained_type in pretrained_types:
                with self.subTest(variant=variant, pretrained_type=pretrained_type):
                    pretrained_num_classes = 21841 if pretrained_type == '22k_distill' else 1000
                    model = timm.create_model(variant, pretrained=True, num_classes=pretrained_num_classes)
                    assert model.head.weight.shape[0] == pretrained_num_classes
                    model = timm.create_model(variant, pretrained=True, pretrained_type=pretrained_type)
                    assert model.head.weight.shape[0] == pretrained_num_classes

    def test_finetune(self):
        pretrained_num_classes = 1000
        finetune_num_classes = 100
        model1 = timm.create_model('tiny_vit_5m_224', pretrained=True, pretrained_type='22kto1k_distill')
        model2 = timm.create_model('tiny_vit_5m_224', pretrained=True, pretrained_type='22kto1k_distill',
            num_classes=finetune_num_classes)
        state_dict_1 = model1.state_dict()
        state_dict_2 = model2.state_dict()
        keys = list(state_dict_1.keys())
        head_keys = ['head.weight', 'head.bias']
        for name in head_keys:
            self.assertEqual(state_dict_1.pop(name).shape[0], pretrained_num_classes)
            self.assertEqual(state_dict_2.pop(name).shape[0], finetune_num_classes)
        for key in keys:
            if key not in head_keys:
                self.assertTrue(torch.equal(state_dict_1[key], state_dict_2[key]))

    def test_forward(self):
        for variant, _ in self.ckpt_names:
            with self.subTest(variant=variant):
                model = timm.create_model(variant)
                img_size = int(variant.split('_')[-1])
                img = torch.randn(1, 3, img_size, img_size)
                out = model(img)
                assert out.shape[-1] == 1000


if __name__ == '__main__':
    unittest.main()
