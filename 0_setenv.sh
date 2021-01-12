#!/bin/bash

export nnUNet_raw_data_base=$PWD/raw_data
export nnUNet_preprocessed=$PWD/preprocessed
export RESULTS_FOLDER=$PWD/trained_models

echo "setting env: "
echo "$nnUNet_raw_data_base"
echo "$nnUNet_preprocessed"
echo "$RESULTS_FOLDER"


