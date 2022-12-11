# Assignment 2 Sensor fusion: filtering disparity images
## Download data:
`cd data` <br/>
`chmod +x load_all_data.sh` <br/>
`./load_all_data.sh` <br/>
`cd ..` <br/>
## Run program with gui
`mkdir build` <br/>
`cd build` <br/>
`cmake ..` <br/>
`make` <br/>
`cd ..` <br/>
`./build/filters --help # To see the options` <br/> 
`./build/filters -H # To run the program` <br/> 

## Visualize pointclouds:
`./visualize.py [-d<Dataset>] [-m[Algorithm]] [--help/-h]` <br/>
**Note:** to produce pointclouds, use `-p` flag while running the program

## Report
Report is in `analaysis.pdf` file