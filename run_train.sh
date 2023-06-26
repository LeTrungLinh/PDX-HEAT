#!/bin/bash

# Read the mode from the user
echo "Enter the mode (1 (Train mode) or 2 (Predict mode) or 3 (Test mode)):"
read mode

# Check the mode and execute the corresponding command
if [ "$mode" = "1" ]; then
    echo "Running Command 1 - Train Mode"
    # Add your command 1 here
    python train.py \
      --exp_dataset outdoor \
      --epochs 900 \
      --lr_drop 850 \
      --batch_size 2 \
      --output_dir ./checkpoints_datnt200/ckpts_heat_sketch_256 \
      --image_size 256 \
      --num_workers 4 \
      --max_corner_num 150 \
      --lambda_corner 0.05 \
      --lr 0.0002 \
      --weight_decay 0.00001 \
      --resume ./checkpoints/ckpts_heat_outdoor_256/checkpoint.pth \
      --run_validation
elif [ "$mode" = "2" ]; then
    echo "Running Command 2 - Predict Mode"
    # Add your command 2 here
    python predict.py \
      --checkpoint_path ./checkpoints_linhlt4/checkpoint_best.pth  \
      --dataset outdoor \
      --image_size 256 \
      --viz_base ./results/viz_heat_outdoor_256
elif [ "$mode" = "3" ]; then
    echo "Running Command 3 - Testing Mode"
    # Add your command 2 here
    python infer.py \
      --checkpoint_path ./checkpoints_datnt200/ckpts_heat_sketch_256/checkpoint.pth  \
      --dataset outdoor \
      --image_size 256 \
      --viz_base ./results/viz_heat_outdoor_256 \
      --save_base ./results/npy_heat_outdoor_256
else
    echo "Invalid mode. Please choose 1 or 2 or 3."
fi

