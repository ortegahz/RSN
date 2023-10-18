import argparse
import logging
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(args):
    img = cv2.imread(args.path_img)

    # session = ort.InferenceSession(args.path_weights, providers=['CPUExecutionProvider'])
    # names_out = [i.name for i in session.get_outputs()]

    model = onnx.load(args.path_weights)
    onnx.checker.check_model(model)
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    # model.graph.output.extend([onnx.ValueInfoProto(name='input.4')])
    session = ort.InferenceSession(model.SerializeToString())
    names_out = [x.name for x in session.get_outputs()]

    # img = (img.astype(np.float32) / 255. - np.array([0.406, 0.456, 0.485])) / np.array([0.225, 0.224, 0.229])
    img = (img.astype(np.float32) - np.array([103.5300, 116.2800, 123.6750])) / np.array([57.3750, 57.1200, 58.3950])
    np.savetxt('/home/manu/tmp/onnx_output_img_wa.txt', img.flatten(), fmt="%f", delimiter="\n")
    img = img.transpose((2, 0, 1))
    batch = np.expand_dims(img, 0)
    batch = batch.astype(np.float32)
    batch = np.ascontiguousarray(batch)

    # outputs = session.run(names_out, {'images': batch})
    # for i, output in enumerate(outputs):
    #     np.savetxt(f'/home/manu/tmp/onnx_outputs_rsn_{i}.txt', output.flatten(), fmt="%f", delimiter="\n")

    outputs = session.run(names_out, {'images': batch})
    outputs = dict(zip(names_out, outputs))
    output_names = list(outputs.keys())
    logging.info(output_names)
    for name_out in outputs.keys():
        np.savetxt(f'/home/manu/tmp/onnx_outputs_rsn_{name_out}.txt', outputs[name_out].flatten(), fmt="%f",
                   delimiter="\n")


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_weights', type=str, default='/home/manu/tmp/iter-96000.onnx')
    parser.add_argument('--path_img', default='/media/manu/samsung/pics/kps_align.bmp')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
