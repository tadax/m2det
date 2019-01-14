#!/bin/bash
wget 'http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz'
tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz
mkdir pretrained
mv resnet_v2_fp32_savedmodel_NHWC/1538687283/variables/* pretrained/
rm -rf resnet_v2_fp32_savedmodel_NHWC/
rm resnet_v2_fp32_savedmodel_NHWC.tar.gz
