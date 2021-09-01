#https://drive.google.com/file/d/1rOaegGwG_zYmCSF32P2We2uRYz7Mp7Jb/view?usp=sharing
FILE_ID="1rOaegGwG_zYmCSF32P2We2uRYz7Mp7Jb"
OUTPUT_NAME='ckpt/VAE/VAE_lr0.001_kldw0.1_MSE0.0134_KLD53.4119.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
