#!/bin/zsh

DIR="dirname \"$(readlink -f \"$0\")\""
CUR=$PWD
cd $DIR/darknet
./darknet detect cfg/yolov3.cfg yolov3.weights $CUR/$1