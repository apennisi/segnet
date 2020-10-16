# SegNet
This repository is SegNet architecture for Semantic Segmentation of Oral Cancer images.

## Usage

### train

`python train.py --save_dir name_of_the_model --train_list /path/to/the/text/file --val_list /path/to/the/text/file  --batch_size n --test_list /path/to/the/text/file`

This step performs both validation and test (apart from the training)

The `epoch_steps` parameter has to be modified according to the number of images and the `batch_size`: epoch_steps = n_images \ batch_size

To generate the images in black and white:

`python test.py --save_dir /path/to/the/directory/ --test_list /path/to/the/text/file --resume /path/to/the/model`

To generate the color images

`python masks.py --save_dir /path/to/the/directory/ --result_dir /path/where/the/black/and/white/images/have/been/saved/ --test_list /path/to/the/text/file`

Each text file has to respect the following standard:
```/absolute/path/to/the/rgb/image_1.png /absolute/path/to/the/mask/image_1.png
/absolute/path/to/the/rgb/image_2.png /absolute/path/to/the/mask/image_2.png
...
/absolute/path/to/the/rgb/image_n.png /absolute/path/to/the/mask/image_n.png```


