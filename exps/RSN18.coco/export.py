#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import logging
import os
from io import BytesIO

import onnx
import torch
import torchvision.transforms as transforms

from config import cfg
from dataset.attribute import load_dataset
from network import RSN


class Exporter:
    def __init__(self, weights, device=0):
        cuda = device != -1 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = RSN(cfg)
        self.model.to(self.device)
        model_file = weights
        if os.path.exists(model_file):
            state_dict = torch.load(
                model_file, map_location=self.device)
            state_dict = state_dict['model']
            self.model.load_state_dict(state_dict)

        self.attr = load_dataset(cfg.DATASET.NAME)
        normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def export_onnx(self, args):
        img = torch.zeros(1, 3, *self.attr.INPUT_SHAPE).to(self.device)
        self.model.eval()
        logging.info(self.model)
        y = self.model(img)  # dry run
        logging.info(y)

        # ONNX export
        try:
            logging.info('\nStarting to export ONNX...')
            export_file = args.weights.replace('.pth', '.onnx')
            with BytesIO() as f:
                torch.onnx.export(self.model, img, f, verbose=False, opset_version=11,
                                  training=torch.onnx.TrainingMode.EVAL,
                                  do_constant_folding=True,
                                  input_names=['images'],
                                  output_names=['outputs'],
                                  dynamic_axes=None)
                f.seek(0)
                # Checks
                onnx_model = onnx.load(f)  # load onnx model
                onnx.checker.check_model(onnx_model)  # check onnx model
            if args.simplify:
                try:
                    import onnxsim
                    logging.info('\nStarting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    logging.info(f'Simplifier failure: {e}')
            onnx.save(onnx_model, export_file)
            logging.info(f'ONNX export success, saved as {export_file}')
        except Exception as e:
            logging.info(f'ONNX export failure: {e}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='/home/manu/tmp/iter-96000.pth', type=str)
    parser.add_argument('--device', default=0, type=int, help='-1 for cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    return parser.parse_args()


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def run(args):
    exporter = Exporter(args.weights, args.device)
    logging.info(exporter)
    exporter.export_onnx(args)


def main():
    args = parse_args()
    set_logging()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
