#https://drive.google.com/file/d/19mnKwrkTW-sSD5Ime-b9fIyuU3RYknB_/view?usp=sharing
FILE_ID="19mnKwrkTW-sSD5Ime-b9fIyuU3RYknB_"
OUTPUT_NAME="SWD.zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id="$FILE_ID -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
unzip $OUTPUT_NAME
mv "SWD" "ckpt/"
rm $OUTPUT_NAME