# TODO: create shell script for running your prototypical network

# Example
#python3 p1.py $1 $2 $3 $4

# IMG_DIR="hw4_data/val"
# IMG_CSV_PATH="hw4_data/val.csv"
# TESTCASE_PATH="hw4_data/val_testcase.csv"
# OUTPUT_PATH="results/pb3_pred.csv" 

IMG_CSV_PATH=$1
IMG_DIR=$2
TESTCASE_PATH=$3
OUTPUT_PATH=$4

python3 src/inference.py \
    -model_path "ckpt/pb1.1/best_loss_1.30687_fc64_5way1shot_euclid.pth"\
    -img_dir $IMG_DIR \
    -img_csv_path $IMG_CSV_PATH \
    -testcase_csv_path $TESTCASE_PATH \
    -output_csv_path $OUTPUT_PATH

# $1: testing images csv file (e.g., hw4_data/val.csv)
# $2: testing images directory (e.g., hw4_data/val)
# $3: path of test case on test set (e.g., hw4_data/val_testcase.csv)
# $4: path of output csv file (predicted labels) (e.g., output/val_pred.csv)



### Problem 4-1-1 ###
# for DIM in 32 64 128 256 512 1024
# do
#     python3 src/train_protonet.py \
#        -fc_dim $DIM
# done

### Problem 4-1-2 ###
# for DIST in 'parametric' 'cosine' 'euclid' 
# do
#     python3 src/train_protonet.py \
#        -train_way 5 \
#        -val_way 5 \
#        -shot 1 \
#        -dest_dir "ckpt/pb1.2" \
#        -distance $DIST
# done

### Problem 4-1-3 ###
