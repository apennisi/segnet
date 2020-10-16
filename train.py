import argparse
from generator import data_gen_small, test_data_generator
from model import segnet
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import img_to_array
from keras import backend as K
import random
import cv2


def dice_coef(y_true, y_pred, smooth=1e-10):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=1000):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description="SegNet LIP dataset")
    parser.add_argument("--save_dir", help="output directory")
    parser.add_argument("--train_list", help="train list path")
    parser.add_argument("--val_list", help="val list path")
    parser.add_argument("--test_list", help="train test list path")
    parser.add_argument("--batch_size", default=3, type=int, help="batch size")
    parser.add_argument("--n_epochs", default=20, type=int, help="number of epoch")
    parser.add_argument("--resume", help="path to the model to resume")
    parser.add_argument(
        "--epoch_steps", default=252, type=int, help="number of epoch step"
    )
    parser.add_argument(
        "--val_steps", default=30, type=int, help="number of valdation step"
    )
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
    parser.add_argument("--optimizer", default="adadelta", type=str, help="oprimizer")
    args = parser.parse_args()

    return args


def main(args):
    # set the necessary list
    train_file = open(args.train_list, "r")
    train_list =  train_file.readlines()
    random.shuffle(train_list)

    val_file = open(args.val_list, "r")
    val_list =  val_file.readlines()

    test_file = open(args.test_list, "r")
    test_list =  test_file.readlines()

    train_gen = data_gen_small(
        train_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )
    val_gen = data_gen_small(
        val_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )

    model = segnet(
        args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode
    )
    print(model.summary())

    model.compile(loss=args.loss, optimizer=Adam(lr=0.001), metrics=["accuracy"])

    if args.resume:
        print("Load Model: " + args.resume)
        model.load_weights(args.resume)

    model.fit_generator(
        train_gen,
        steps_per_epoch=args.epoch_steps,
        epochs=args.n_epochs,
        validation_data=val_gen,
        validation_steps=args.val_steps,
    )

    model.save_weights(args.save_dir + str(args.n_epochs) + ".hdf5")
    print("save weight done..")

    test_gen = test_data_generator(
        test_list,
        #args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )
    model.evaluate(test_gen)


if __name__ == "__main__":
    args = argparser()
    main(args)
