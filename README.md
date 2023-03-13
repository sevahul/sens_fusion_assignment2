# Assignment 2 Sensor fusion: Filtering and Upsampling disparity images
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
Report with results analysis and execution results is in `analaysis.pdf` file.
The repo runs out of the box using cached data for generating plots (analysis.ipynb)

## GUI example
If you run the program with GUI, it would look like that, allowing to see how different parameters influence the Filtered/Upsampled Image: <br>
<img src="https://github.com/sevagul/sens_fusion_assignment2/blob/main/output/gui.png">
