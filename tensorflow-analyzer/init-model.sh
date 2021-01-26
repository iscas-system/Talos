NUM_GPUS=1
DATA_DIR=/root/github/tensorflow-dataset
MODEL_DIR=/root/github/tensorflow-model
export PYTHONPATH=$PYTHONPATH:/root/github/Talos/models

# Mnist
# python3 mnist_main.py   --model_dir=$MODEL_DIR   --data_dir=$DATA_DIR   --train_epochs=1   --distribution_strategy=one_device   --num_gpus=$NUM_GPUS   --download