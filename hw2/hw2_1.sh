# ResNet18
# https://drive.google.com/file/d/1hdYWIQGlKxdL1iN5FL3LJRQ9V3Por-us/view?usp=sharing
FILE_ID="1hdYWIQGlKxdL1iN5FL3LJRQ9V3Por-us"
OUTPUT_NAME='ckpt/ResNet18.ckpt'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt


python3 src/inference2_1.py \
    -model_path $OUTPUT_NAME \
    -batch_size 32 \
    -data_path $1 \
    -dest_dir $2 
    
