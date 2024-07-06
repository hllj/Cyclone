#!/bin/bash

# COCO2017

# export DATA_DIR=/mnt/disks/dev/data/images

unzip -j ${DATA_DIR}/coco2017/train2017.zip -d ${DATA_DIR}/coco2017/train2017
unzip -j ${DATA_DIR}/coco2017/val2017.zip -d ${DATA_DIR}/coco2017/val2017
rm ${DATA_DIR}/coco2017/train2017.zip ${DATA_DIR}/coco2017/val2017.zip
mv ${DATA_DIR}/coco2017 ${DATA_DIR}/coco
