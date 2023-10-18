import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from config import cfg
from network import RSN

from dataset.attribute import load_dataset
from lib.utils.transforms import flip_back
from lib.utils.transforms import get_affine_transform


class Inferer:
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

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.attr.PIXEL_STD, h * 1.0 / self.attr.PIXEL_STD],
                         dtype=np.float32)

        return center, scale

    def inference(self, img, dets):
        if self.attr.COLOR_RGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs, scores, centers, scales = [], [], [], []
        for *bbox, score in dets:
            center, scale = self._bbox_to_center_and_scale(bbox)
            rotation = 0
            scale[0] *= (1. + self.attr.TEST.X_EXTENTION)
            scale[1] *= (1. + self.attr.TEST.Y_EXTENTION)
            # fit the ratio
            if scale[0] > self.attr.WIDTH_HEIGHT_RATIO * scale[1]:
                scale[1] = scale[0] * 1.0 / self.attr.WIDTH_HEIGHT_RATIO
            else:
                scale[0] = scale[1] * 1.0 * self.attr.WIDTH_HEIGHT_RATIO
            trans = get_affine_transform(center, scale, rotation, self.attr.INPUT_SHAPE)

            img_wa = cv2.warpAffine(img, trans, (int(self.attr.INPUT_SHAPE[1]), int(self.attr.INPUT_SHAPE[0])),
                                    flags=cv2.INTER_LINEAR)
            # cv2.imwrite('/home/manu/tmp/rsn.bmp', img_wa)
            np.savetxt('/home/manu/tmp/pytorch_output_img_wa.txt', img_wa.flatten(), fmt="%f", delimiter="\n")
            if self.transform:
                img_wa = self.transform(img_wa)

            imgs.append(img_wa)
            scores.append(score)
            centers.append(center)
            scales.append(scale)

        self.model.eval()
        results = list()
        cpu_device = torch.device("cpu")
        imgs = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            outputs = self.model(imgs)
            np.savetxt('/home/manu/tmp/pytorch_outputs_rsn.txt',
                       outputs.detach().cpu().numpy().flatten(),
                       fmt="%f", delimiter="\n")
            outputs = outputs.to(cpu_device).numpy()

            if cfg.TEST.FLIP:
                imgs_flipped = np.flip(imgs.to(cpu_device).numpy(), 3).copy()
                imgs_flipped = torch.from_numpy(imgs_flipped).to(self.device)
                outputs_flipped = self.model(imgs_flipped)
                outputs_flipped = outputs_flipped.to(cpu_device).numpy()
                outputs_flipped = flip_back(
                    outputs_flipped, cfg.DATASET.KEYPOINT.FLIP_PAIRS)
                outputs = (outputs + outputs_flipped) * 0.5
        centers = np.array(centers)
        scales = np.array(scales)
        preds, maxvals = self._get_results(outputs, centers, scales,
                                           cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)
        kp_scores = maxvals.squeeze(axis=-1).mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()
            score = scores[i] * kp_scores[i]

            results.append(dict(category_id=1,
                                keypoints=keypoints,
                                score=score))
        return results

    def draw_results(self, img, results):
        for i, result in enumerate(results):
            score = results[i]['score']
            joints = np.array(results[i]['keypoints']).reshape((self.attr.KEYPOINT.NUM, 3))
            pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                     [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
            color = np.random.randint(0, 256, (self.attr.KEYPOINT.NUM, 3)).tolist()

            for i in range(self.attr.KEYPOINT.NUM):
                if i != 9 and i != 10:
                    continue
                color = (255, 0, 0) if i == 9 else (0, 255, 0)
                if joints[i, 0] > 0 and joints[i, 1] > 0:
                    # cv2.circle(img, tuple(joints[i, :2].astype(int)), 2, tuple(color[i]), 2)
                    cv2.circle(img, tuple(joints[i, :2].astype(int)), 2, color, 2)
            # if score:
            #     cv2.putText(img, f'{score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            #                 (128, 255, 0), 2)

            def draw_line(img, p1, p2):
                c = (0, 0, 255)
                if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                    cv2.line(img, tuple(p1), tuple(p2), c, 2)

            # for pair in pairs:
            #     draw_line(img, joints[pair[0] - 1, :2].astype(int), joints[pair[1] - 1, :2].astype(int))

        return img

    @staticmethod
    def _get_results(outputs, centers, scales, kernel=11, shifts=[0.25]):
        scales *= 200
        nr_img = outputs.shape[0]
        preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
        maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
        for i in range(nr_img):
            score_map = outputs[i].copy()
            score_map = score_map / 255 + 0.5
            kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
            scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
            border = 10
            dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
                           cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
            dr[:, border: -border, border: -border] = outputs[i].copy()
            for w in range(cfg.DATASET.KEYPOINT.NUM):
                # np.savetxt('/home/manu/tmp/pytorch_output_dr_%s.txt' % w, dr[w].flatten(), fmt="%f", delimiter="\n")
                dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
                np.savetxt('/home/manu/tmp/pytorch_output_dr_%s.txt' % w, dr[w].flatten(), fmt="%f", delimiter="\n")
            for w in range(cfg.DATASET.KEYPOINT.NUM):
                for j in range(len(shifts)):
                    if j == 0:
                        lb = dr[w].argmax()
                        y, x = np.unravel_index(lb, dr[w].shape)
                        dr[w, y, x] = 0
                        x -= border
                        y -= border
                    lb = dr[w].argmax()
                    py, px = np.unravel_index(lb, dr[w].shape)
                    dr[w, py, px] = 0
                    px -= border + x
                    py -= border + y
                    ln = (px ** 2 + py ** 2) ** 0.5
                    if ln > 1e-3:
                        x += shifts[j] * px / ln
                        y += shifts[j] * py / ln
                x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
                y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
                kps[w] = np.array([x * 4 + 2, y * 4 + 2])
                scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                                         int(round(x) + 1e-9)]
                fn_txt = '/home/manu/tmp/results_pytorch_rsn_mid.txt'
                if os.path.exists(fn_txt):
                    os.remove(fn_txt)
                for kp, score in zip(kps, scores):
                    with open(fn_txt, 'a') as f:
                        f.write(f'{kp[0]} {kp[1]} {score[0]} \n')
            # aligned or not ...
            kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
                        centers[i][0] - scales[i][0] * 0.5
            kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
                        centers[i][1] - scales[i][1] * 0.5
            preds[i] = kps
            maxvals[i] = scores

        return preds, maxvals


def draw_dets(img, dets, color=(0, 255, 255)):
    for x, y, w, h, score in dets:
        cv2.rectangle(img, (int(x), int(y)),
                      (int(x + w), int(y + h)), color,
                      thickness=5, lineType=cv2.LINE_AA)
        cv2.putText(img, f'{score:.2f}', (int(x), int(y) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        default='/media/manu/samsung/pics/kps.bmp',
                        # default='/home/manu/nfs/rv1126/install/rknn_yolov5_demo/model/player_1280.bmp',
                        type=str)
    parser.add_argument('--weights', default='/home/manu/tmp/iter-96000.pth', type=str)
    parser.add_argument('--device', default=0, type=int, help='-1 for cpu')
    return parser.parse_args()


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def main():
    args = parse_args()
    set_logging()
    logging.info(args)
    inferer = Inferer(args.weights, args.device)

    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    # cv2.imwrite('/home/manu/tmp/rsn_org.bmp', img)
    dets = np.array([[153.53, 231.12, 270.17, 403.95, 0.3091]])  # [x, y, w, h, score]
    # dets = np.array([[825., 679., 111.1, 244.2, 0.92078]])  # [x, y, w, h, score]

    # draw_dets(img, dets)

    results = inferer.inference(img, dets)
    logging.info(results)

    fn_txt = '/home/manu/tmp/results_pytorch_rsn.txt'
    if os.path.exists(fn_txt):
        os.remove(fn_txt)
    joints = np.array(results[0]['keypoints']).reshape((17, 3))
    for kp in joints:
        with open(fn_txt, 'a') as f:
            f.write(f'{kp[0]} {kp[1]} {kp[2]} \n')

    img = inferer.draw_results(img, results)
    cv2.imshow('results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
