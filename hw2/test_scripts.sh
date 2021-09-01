#pip install -r requirements.txt

bash hw2_1.sh "../hw2_data/p1_data/test_50" "../results/"
bash hw2_2.sh "../hw2_data/p2_data/test" "../results/FCN32s"
bash hw2_2_best.sh "../hw2_data/p2_data/test" "../results/FCN8s"

echo "Evaluating hw2_1"
python3 src/checker/cls_checker.py -pred_path "../results/test_pred.csv" -label_path "../hw2_data/p1_data/test_gt.csv"

echo "Evaluating hw2_2"
python3 src/checker/mean_iou_evaluate.py --pred "../hw2_data/p2_data/validation" --labels "../results/FCN32s"

echo "Evaluating hw2_2_best"
python3 src/checker/mean_iou_evaluate.py --pred "../hw2_data/p2_data/validation" --labels "../results/FCN8s"
