#!/bin/bash
pwd=$pwd
cd submodules/mpeg-pcc-tmc13
if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake ..
make 
cd $pwd