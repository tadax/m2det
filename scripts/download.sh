#!/bin/bash

wget 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz'
tar -xvzf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz

if [ ! -d "weights" ]; then
    mkdir model
fi

mv vgg_16.ckpt weights/vgg_16.ckpt
