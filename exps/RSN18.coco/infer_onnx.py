import argparse
import logging
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(args):
    img = cv2.imread(args.path_img)

    session = ort.InferenceSession(args.path_weights, providers=['CPUExecutionProvider'])
    names_out = [i.name for i in session.get_outputs()]

    # img = (img.astype(np.float32) / 255. - np.array([0.406, 0.456, 0.485])) / np.array([0.225, 0.224, 0.229])
    img = (img.astype(np.float32) - np.array([103.5300, 116.2800, 123.6750])) / np.array([57.3750,  57.1200,  58.3950])
    img = img.transpose((2, 0, 1))
    batch = np.expand_dims(img, 0)
    batch = batch.astype(np.float32)
    batch = np.ascontiguousarray(batch)

    outputs = session.run(names_out, {'images': batch})
    for i, output in enumerate(outputs):
        np.savetxt(f'/home/manu/tmp/onnx_outputs_{i}.txt', output.flatten(), fmt="%f", delimiter="\n")


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_weights', type=str, default='/home/manu/tmp/iter-96000.onnx')
    parser.add_argument('--path_img', default='/media/manu/samsung/pics/rsn.bmp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
