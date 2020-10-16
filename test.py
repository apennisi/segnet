import argparse
from generator import data_gen_small, test_data_generator
from model import segnet
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import img_to_array
import random
import cv2
import numpy as np

color_table = [0, 128, 255]

def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet LIP dataset")
    parser.add_argument("--save_dir", help="output directory")
    parser.add_argument("--test_list", help="train test list path")
    parser.add_argument("--resume", help="path to the model to resume")
    parser.add_argument("--n_labels", default=3, type=int, help="Number of label")
    parser.add_argument(
        "--input_shape", default=(512, 512, 3), help="Input images shape"
    )
    parser.add_argument("--kernel", default=5, type=int, help="Kernel size")
    parser.add_argument(
        "--pool_size", default=(2, 2), help="pooling and unpooling size"
    )
    parser.add_argument(
        "--output_mode", default="softmax", type=str, help="output activation"
    )
    parser.add_argument(
        "--loss", default="categorical_crossentropy", type=str, help="loss function"
    )
    args = parser.parse_args()

    return args

def image_list(lists):
    imgs = []
    for i in range(len(lists)):
        line = lists[i]
        name = line.split(" ")[0]
        imgs.append(name.rstrip("\n"))
    return imgs

def convert_image(img, size):
    resized_img = cv2.resize(img, size)
    array_img = img_to_array(resized_img) / 255
    imgs = []
    imgs.append(array_img)
    imgs = np.array(imgs)
    return imgs

def color_image(result, size, n_classes, color_table):
    result = result.reshape((size[0], size[1], n_classes)).argmax(axis=2)
    new_image = np.zeros(size)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if result[i, j] == 1:
                new_image[i, j] = 128
            elif result[i, j] == 2:
                new_image[i, j] = 255
            else:
                pass
    return new_image

def main(args):
    # set the necessary list
    test_file = open(args.test_list, "r")
    test_list =  test_file.readlines()

    model = segnet(
        args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode
    )
    print(model.summary())

    model.compile(loss=args.loss, optimizer=Adam(lr=0.0001), metrics=["accuracy"])

    
    print("Load Model: " + args.resume)
    model.load_weights(args.resume)

    list_images = image_list(test_list)

    size = (args.input_shape[0], args.input_shape[1])

    for image in list_images:
        print(image)
        original_img = cv2.imread(image)[:, :, ::-1]
        converted_img = convert_image(original_img, size)
        result = model.predict(converted_img)[0]
        colored_image = color_image(result, size, args.n_labels, color_table)
        last_slash = image.rfind('/')
        name = image[last_slash+1:]
        cv2.imwrite(args.save_dir + "/" + name, colored_image )

if __name__ == "__main__":
    args = argparser()
    main(args)
