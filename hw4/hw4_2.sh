# TODO: create shell script for running your data hallucination model

# Example
#python3 p2.py $1 $2 $3 $4

# IMG_DIR="hw4_data/val"
# IMG_CSV_PATH="hw4_data/val.csv"
# TESTCASE_PATH="hw4_data/val_testcase.csv"
# OUTPUT_PATH="results/pb3_pred.csv" 

IMG_CSV_PATH=$1
IMG_DIR=$2
TESTCASE_PATH=$3
OUTPUT_PATH=$4

python3 src/inference.py -hall \
    -model_path "ckpt/pb2.1/best_loss_1.28753_fc64_M150.pth"\
    -fc_dim 64 \
    -M 150 \
    -img_dir $IMG_DIR \
    -img_csv_path $IMG_CSV_PATH \
    -testcase_csv_path $TESTCASE_PATH \
    -output_csv_path $OUTPUT_PATH


### Problem 4-2-1 ###
# for DIM in 64 128 256
# do
#     for M in 20 50 100 150 200
#     do
#         python3 src/train_hall.py \
#             -fc_dim $DIM  \
#             -M $M
#     done
# done


## Problem 4-2-3 ###
# for M in 10 50 100
# do
#     python3 src/train_hall.py \
#         -train_way 5 \
#         -val_way 5 \
#         -dest_dir "ckpt/pb2.3" \
#         -M $M
# done



