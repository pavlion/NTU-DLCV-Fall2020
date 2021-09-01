# FCN32s
# https://drive.google.com/file/d/11_1KMQlH7nzFggxqYWD1o9swovBWWHoW/view?usp=sharing
FILE_ID="11_1KMQlH7nzFggxqYWD1o9swovBWWHoW"
OUTPUT_NAME='ckpt/FCN32s.ckpt'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt


python3 src/inference2_2.py \
    -batch_size 4 \
    -model_path $OUTPUT_NAME \
    -data_path $1 \
    -dest_dir $2 





