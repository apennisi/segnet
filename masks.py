import argparse
import random
import cv2
import numpy as np
import os

color_table = [0, 128, 255]

def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet LIP dataset")
    parser.add_argument("--save_dir", help="output directory")
    parser.add_argument("--result_dir", help="result directory")
    parser.add_argument("--test_list", help="train test list path")
    parser.add_argument(
        "--input_shape", default=(512, 512, 3), help="Input images shape"
    )
    args = parser.parse_args()

    return args

def image_list(lists):
    imgs = []
    for i in range(len(lists)):
        line = lists[i]
        name = line.split(" ")[1]
        imgs.append(name.rstrip("\n"))
    return imgs


def color_image(image, size, color_table):
    new_image = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if image[i, j] == 1:
                new_image[i, j] = 128
            elif image[i, j] == 2:
                new_image[i, j] = 255
            else:
                pass
    return new_image


def main(args):
    # set the necessary list
    test_file = open(args.test_list, "r")
    test_list =  test_file.readlines()

    mask_list = image_list(test_list)
    results = os.listdir(args.result_dir)
    results = [args.result_dir + "/" + name for name in results]

    results = sorted(results)
    mask_list = sorted(mask_list)

    for gt, result in zip(mask_list, results):
        print(gt, result)
        gt_img = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (512, 512))
        result_image = cv2.imread(result, cv2.IMREAD_GRAYSCALE)
        gt_img = color_image(gt_img, (512, 512), color_table)
        new_image = np.zeros(args.input_shape)
        for i in range(512):
            for j in range(512):
                if gt_img[i, j] == 255 and result_image[i, j] == 255:
                    new_image[i, j, :] = (0, 255, 0)
                elif gt_img[i, j] == 128 and result_image[i, j] == 128:
                    new_image[i, j, :] = (120, 120, 120)
                elif gt_img[i, j] == 255 and result_image[i, j] != 255:
                    new_image[i, j, :] = (0, 0, 255)
                elif gt_img[i, j] == 128 and result_image[i, j] != 128:
                    new_image[i, j, :] = (255, 0, 0)
                elif gt_img[i, j] != 0 and result_image[i, j] != 0:
                    new_image[i, j, :] = (90, 90, 90)
                else:
                    pass
        
        last_slash = gt.rfind('/')
        name = gt[last_slash+1:]
        cv2.imwrite(args.save_dir + "/" + name, new_image)

if __name__ == "__main__":
    args = argparser()
    main(args)