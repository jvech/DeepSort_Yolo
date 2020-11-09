#!/usr/bin/bash
FILE_ID="1bt-qbLLqY97UfrWp99hlEIUFcKEgKvDX" 
FILENAME="yolov3_weights.zip"

echo "Downloading the weights"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILE_ID}" -o ${FILENAME}
rm ./cookie

echo "Decompressing the weights"
unzip -d ./data ${FILENAME}
rm ${FILENAME}
