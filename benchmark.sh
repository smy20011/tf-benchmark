#!/bin/bash

### Change this to match the repo root dir ###
RPATH="/benchmarks"

#Ouput results to home dir
RESULTS="/output/tf_cnn_bench_results.txt"
#Full path
FPATH="$RPATH/scripts/tf_cnn_benchmarks"

#Run the benches
python "$FPATH/tf_cnn_benchmarks.py" --num_gpus=1 --batch_size=64 --variable_update=parameter_server --optimizer=sgd --model=resnet50 --local_parameter_device=gpu | tee -a "$RESULTS"
sleep 5
python "$FPATH/tf_cnn_benchmarks.py" --num_gpus=1 --batch_size=32 --variable_update=parameter_server --optimizer=sgd --model=resnet152 --local_parameter_device=gpu | tee -a "$RESULTS"
sleep 5
python "$FPATH/tf_cnn_benchmarks.py" --num_gpus=1 --batch_size=512 --variable_update=parameter_server --optimizer=sgd --model=alexnet --local_parameter_device=gpu | tee -a "$RESULTS"
sleep 5
python "$FPATH/tf_cnn_benchmarks.py" --num_gpus=1 --batch_size=32 --variable_update=parameter_server --optimizer=sgd --model=vgg16 --local_parameter_device=gpu | tee -a "$RESULTS"
sleep 5
python "$FPATH/tf_cnn_benchmarks.py" --num_gpus=1 --batch_size=64 --variable_update=parameter_server --optimizer=sgd --model=inception3 --local_parameter_device=gpu | tee -a "$RESULTS"
