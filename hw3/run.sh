bash hw3_p1.sh 'results/VAE.png'
bash hw3_p2.sh 'results/GAN.png'

MODEL_TYPE='DANN'

bash hw3_p3.sh '../hw3_data/digits/svhn/test' 'svhn' 'results/'$MODEL_TYPE'_mnistm_svhn_test_pred.csv' 
bash hw3_p3.sh '../hw3_data/digits/usps/test' 'usps' 'results/'$MODEL_TYPE'_svhn_usps_test_pred.csv' 
bash hw3_p3.sh '../hw3_data/digits/mnistm/test' 'mnistm' 'results/'$MODEL_TYPE'_usps_mnistm_test_pred.csv' 


echo "mnistm_svhn"
python hw3_eval.py 'results/'$MODEL_TYPE'_mnistm_svhn_test_pred.csv' '../hw3_data/digits/svhn/test.csv'

echo "svhn_usps"
python hw3_eval.py 'results/'$MODEL_TYPE'_svhn_usps_test_pred.csv' '../hw3_data/digits/usps/test.csv'

echo "usps_mnistm"
python hw3_eval.py 'results/'$MODEL_TYPE'_usps_mnistm_test_pred.csv' '../hw3_data/digits/mnistm/test.csv'


MODEL_TYPE='SWD'

bash hw3_p4.sh '../hw3_data/digits/svhn/test' 'svhn' 'results/'$MODEL_TYPE'_mnistm_svhn_test_pred.csv' 
bash hw3_p4.sh '../hw3_data/digits/usps/test' 'usps' 'results/'$MODEL_TYPE'_svhn_usps_test_pred.csv' 
bash hw3_p4.sh '../hw3_data/digits/mnistm/test' 'mnistm' 'results/'$MODEL_TYPE'_usps_mnistm_test_pred.csv' 

echo "mnistm_svhn"
python hw3_eval.py 'results/'$MODEL_TYPE'_mnistm_svhn_test_pred.csv' '../hw3_data/digits/svhn/test.csv'

echo "svhn_usps"
python hw3_eval.py 'results/'$MODEL_TYPE'_svhn_usps_test_pred.csv' '../hw3_data/digits/usps/test.csv'

echo "usps_mnistm"
python hw3_eval.py 'results/'$MODEL_TYPE'_usps_mnistm_test_pred.csv' '../hw3_data/digits/mnistm/test.csv'

