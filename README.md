# M2Det

An implementation of
[_Q Zhao et al., "M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network", 2019_](
https://arxiv.org/pdf/1811.04533.pdf) using TensorFlow.

## Results

COCO 2017 dataset is used for training.

![](data/yolov2_result.jpg)

![](data/yolo_result.jpg)

![](data/innsbruck_result.png)

## Performance

To be released.


## Usage

### Requirements

- Python 3.6
- TensorFlow 1.8

### Model

:warning: Notice: it's work in progress.

You can download the trained model [[weights_2019-01-31](https://dl.dropboxusercontent.com/s/1lige7lokxsu4tn/weights_2019-01-31.tar.gz?dl=0)].

### Run Demo

```
$ python demo.py --inputs <image_path> --model_path <model_path>
```

### How to Train

#### I. Prepare dataset

Download COCO (2017) dataset from [http://cocodataset.org](http://cocodataset.org) 
and process them:

```
$ python mscoco/process.py \
--image_dir <image_dir> \
--annotation_path <annotation_dir> \
--output_dir <output_dir>
```

#### II. Download pretrained model

```
$ cd weights/
$ bash download.sh
```

#### III. Train model

```
$ python train.py --image_dir <image_dir> --label_dir <label_dir>
```

## Note

### With vs. without ImageNet pre-training

ImaneNet pre-training is not used.

cf. [Kaiming He et al., "Rethinking ImageNet Pre-training", 2018](https://arxiv.org/pdf/1811.08883.pdf)


### Learning rate scheduling

To be written.


## License

MIT License

Copyright (c) 2019 tadax

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
