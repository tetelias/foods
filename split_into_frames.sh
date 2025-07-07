mkdir -p data/video21
ffmpeg -y -i "data/2_1.MOV"  "data/video21/%05d.jpeg"
mkdir -p data/video31
ffmpeg -y -i "data/3_1.MOV"  "data/video31/%05d.jpeg"
mkdir -p data/video32
ffmpeg -y -i "data/3_2.MOV"  "data/video32/%05d.jpeg"
mkdir -p data/video4
ffmpeg -y -i "data/4.MOV"  "data/video4/%05d.jpeg"