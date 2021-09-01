# FCN8s
# https://drive.google.com/file/d/13KD8Q3_QdwUwLPQ8H57OdzAF_s3Iy6u0/view?usp=sharing
FILE_ID="13KD8Q3_QdwUwLPQ8H57OdzAF_s3Iy6u0"
OUTPUT_NAME='ckpt/FCN8s.ckpt'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt

python3 src/inference2_2.py \
    --improved \
    -batch_size 4 \
    -model_path $OUTPUT_NAME \
    -data_path $1 \
    -dest_dir $2 



