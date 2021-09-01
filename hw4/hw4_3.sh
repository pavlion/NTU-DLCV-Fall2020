# TODO: create shell script for running your improved data hallucination model


# IMG_DIR="hw4_data/val"
# IMG_CSV_PATH="hw4_data/val.csv"
# TESTCASE_PATH="hw4_data/val_testcase.csv"
# OUTPUT_PATH="results/pb3_pred.csv" 

IMG_CSV_PATH=$1
IMG_DIR=$2
TESTCASE_PATH=$3
OUTPUT_PATH=$4
TRAIN_IMG_CSV_PATH=$5
TRAIN_IMG_DIR=$6

# Example
#python3 p3.py $1 $2 $3 $4 $5 $6
python3 src/inference.py -hall -ifi\
    -model_path "ckpt/pb3.1/best_acc_48.6667_5way1shot_fc64_M20.pth"\
    -fc_dim 64 \
    -img_dir $IMG_DIR \
    -img_csv_path $IMG_CSV_PATH \
    -testcase_csv_path $TESTCASE_PATH \
    -output_csv_path $OUTPUT_PATH



### Problem 4-3-1 ###
# for DIM in 64 128 256
# do
#     for M in 5 10 50 100 150 200
#     do
#         python3 src/train_IFI.py \
#             -fc_dim $DIM  \
#             -M $M
#     done
# done

