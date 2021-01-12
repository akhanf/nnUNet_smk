#!/bin/bash

export nnUNet_raw_data_base=$PWD/raw_data
export nnUNet_preprocessed=$TMPDIR/preprocessed
export RESULTS_FOLDER=$PWD/trained_models
export nnUNet_n_proc_DA=12

echo "setting env: "
echo "$nnUNet_raw_data_base"
echo "$nnUNet_preprocessed"
echo "$RESULTS_FOLDER"


