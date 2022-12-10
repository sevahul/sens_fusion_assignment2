mkdir $1
cd $1
wget -nc  https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/$1/Illum1/Exp0/view0.png
wget -nc  https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/$1/Illum1/Exp0/view1.png
wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/$1/disp1.png
cd ..
