##################
### Submission ###
##################
bash get_SWD_model.sh
python3 src/inference_DANN.py \
    -model_type 'SWD' \
    -batch_size 256 \
    -img_dir $1 \
    -target_domain $2 \
    -dest_path $3
rm -rf 'ckpt/SWD'
    
###################
### Traininging ###
###################
# python src/train_SWD.py \
#     -batch_size 256 \
#     --M 256\
#     --lr 5e-5 \
#     -epoch 20 \
#     -source_domain 'mnistm' \
#     -target_domain 'svhn'


# source only
# python src/train_SWD.py \
#     -batch_size 128 \
#     -epoch 15 \
#     -source_domain 'usps' \
#     -target_domain 'mnistm' \
#     --M 128 \
#     --lr 1e-3 --test

# # source only
# python src/train_SWD.py \
#     -batch_size 128 \
#     -epoch 15 \
#     -source_domain 'mnistm' \
#     -target_domain 'svhn' \
#     --M 128 \
#     --lr 1e-3 --test

# # source only
# python src/train_SWD.py \
#     -batch_size 128 \
#     -epoch 15 \
#     -source_domain 'svhn' \
#     -target_domain 'usps' \
#     --M 128 \
#     --lr 1e-3 --test  



###################
### Inferencing ###
### Train on src###
### Test on trg ###
###################
# python src/train_DANN.py \
#     --no_da \
#     --test \
#     -ckpt_path "ckpt/DANN_lr0.0001_usps-usps/best_loss.pth" \
#     -source_domain 'svhn' \
#     -target_domain 'mnistm'

# python src/train_DANN.py \
#     --no_da \
#     --test \
#     -ckpt_path "ckpt/DANN_lr0.0001_svhn-svhn/best_loss.pth" \
#     -source_domain 'svhn' \
#     -target_domain 'usps'

# python src/train_DANN.py \
#     --no_da \
#     --test \
#     -ckpt_path "ckpt/DANN_lr0.0001_mnistm-mnistm/best_loss.pth" \
#     -source_domain 'svhn' \
#     -target_domain 'svhn'




#################
### Inference ###
#################

# BATCH_SIZE=256
# TARGET='mnistm'
# python src/inference_DANN.py \
#     -batch_size $BATCH_SIZE \
#     -img_dir 'hw3_data/digits/'${TARGET}'/test' \
#     -target_domain $TARGET \
#     -dest_dir 'results/DANN/'$TARGET

# TARGET='usps'
# python src/inference_DANN.py \
#     -batch_size $BATCH_SIZE \
#     -img_dir 'hw3_data/digits/'${TARGET}'/test' \
#     -target_domain $TARGET \
#     -dest_dir 'results/DANN/'$TARGET

# TARGET='svhn'
# python src/inference_DANN.py \
#     -batch_size $BATCH_SIZE \
#     -img_dir 'hw3_data/digits/'${TARGET}'/test' \
#     -target_domain $TARGET \
#     -dest_dir 'results/DANN/'$TARGET


##########################
### Calculate Accuracy ###
##########################

# python src/checker/cls_checker.py \
#     -label_path 'hw3_data/digits/mnistm/test.csv' \
#     -pred_path 'results/DANN/mnistm/test_pred.csv'

# python src/checker/cls_checker.py \
#     -label_path 'hw3_data/digits/usps/test.csv' \
#     -pred_path 'results/DANN/usps/test_pred.csv'

# python src/checker/cls_checker.py \
#     -label_path 'hw3_data/digits/svhn/test.csv' \
#     -pred_path 'results/DANN/svhn/test_pred.csv'


