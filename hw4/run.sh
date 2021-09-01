IMG_CSV_PATH="../hw4_data/val.csv"
IMG_DIR="../hw4_data/val"
TESTCASE_PATH="../hw4_data/val_testcase.csv"
GT_PATH="../hw4_data/val_testcase_gt.csv"

# TESTCASE_PATH="hw4_data/test_testcase.csv"
# GT_PATH="hw4_data/test_testcase_gt.csv"
bash hw4_download.sh

bash hw4_1.sh $IMG_CSV_PATH $IMG_DIR $TESTCASE_PATH "results/pb1_pred.csv"  
python3 eval.py "results/pb1_pred.csv" $GT_PATH

bash hw4_2.sh $IMG_CSV_PATH $IMG_DIR $TESTCASE_PATH "results/pb2_pred.csv"  
python3 eval.py "results/pb2_pred.csv" $GT_PATH

bash hw4_3.sh $IMG_CSV_PATH $IMG_DIR $TESTCASE_PATH "results/pb3_pred.csv" \
    "../hw4_data/train.csv" "../hw4_data/train"
python3 eval.py "results/pb3_pred.csv" $GT_PATH