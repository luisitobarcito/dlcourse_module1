#!/bin/bash

## download cifar10 dataset from Krivshensky site

DOWNLOAD_DIR=../data/cifar/

if [ !  -d "$DOWNLOAD_DIR" ]; then
    # Create cifar directory
    mkdir "$DOWNLOAD_DIR"
fi

cd "$DOWNLOAD_DIR"
wget -v https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar -xvf cifar-10-python.tar.gz
