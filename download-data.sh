#!/bin/sh

# Script to download the competition data and extract it into the 
# data directory 

# Download the data 
kaggle competitions download -c playground-series-s3e3

mkdir -p data/

unzip playground-series-s3e3.zip -d data 

exit 0 



