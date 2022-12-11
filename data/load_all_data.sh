DATASETS="Art Books Dolls Laundry Moebius Reindeer"
for NAME in ${DATASETS}; do
    mkdir -p ${NAME}
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/disp1.png -O ${NAME}/disparity.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/Illum1/Exp0/view1.png -O ${NAME}/intensity.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/dmin.txt -O ${NAME}/dmin.txt;
    
    wget -nc http://server.seva-hul.com/datasets/SensFusion/disp_DP_${NAME}.png -O ${NAME}/disp_DP.png;
done
DATASETS="Aloe Baby1 Bowling1 Cloth1 Flowerpots Midd1"
for NAME in ${DATASETS}; do
    mkdir -p ${NAME}
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2006/FullSize/${NAME}/disp1.png -O ${NAME}/disparity.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2006/FullSize/${NAME}/Illum1/Exp0/view1.png -O ${NAME}/intensity.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2006/FullSize/${NAME}/dmin.txt -O ${NAME}/dmin.txt;

    wget -nc http://server.seva-hul.com/datasets/SensFusion/disp_DP_${NAME}.png -O ${NAME}/disp_DP.png;
done
