import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import cv2
import pandas as pd
import json
import time
import xarray as xr
from collections import OrderedDict
# import custom metrics
try:
    from analysis.compare_disparities import NCC, MSE, SSIM
except:
    from compare_disparities import NCC, MSE, SSIM

# define metrics, datasets, methods and parameters ranges
metrics_dict = {
    "SSIM": lambda x, y: SSIM(x, y, True),
    "MSE_inverted": lambda x, y: MSE(x, y, normed=True, inverted=True),
    "NCC": lambda x, y: NCC(x, y)
}
try:
    datasets = next(os.walk('data'))[1]
    if len(datasets) < 6:
        raise Exception("less then 6 datasets")
except:
    datasets = ['Art', 'Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer', 'Aloe', 'Baby1', 'Bowling1', 'Cloth1', 'Flowerpots', 'Midd1']
methods_upsample = ["Iter", "JBU"]
methods = ["JB", "Bilet"]
method_first = "DP"
w_sizes = np.arange(3, 25, 2).tolist()
srs = np.arange(3, 50, 2).tolist()
gsfs = np.round(np.arange(0.1, 3, 0.2), 1).tolist()

params = OrderedDict([
    ("w_sizes", w_sizes),
    ("srs", srs),
    ("gsfs", gsfs)
])
default_params = OrderedDict([
    ("w_sizes", 10),
    ("gsfs", 1.5),
    ("srs", 25)
])
params_short ={
    "w_sizes": "w",
    "gsfs": "gsf",
    "srs": "sr"
}

## reading and writing existing cached results

# for metrics values
metrics_dict_path = os.path.join("analysis", "metrics.json")
def read_metrics():
    if os.path.isfile(metrics_dict_path):
        with open(metrics_dict_path) as f:
            return json.load(f)
    else:
        return dict() 

def write_metrics(metrics_dict):
    with open(metrics_dict_path, 'w') as convert_file:
        convert_file.write(json.dumps(metrics_dict, indent=4))

# for execution times
times_dict_path = os.path.join("analysis", "times.json")
def read_times():
    if os.path.isfile(times_dict_path):
        with open(times_dict_path) as f:
            return json.load(f)
    else:
        return dict() 

def write_times(times_dict):
    with open(times_dict_path, 'w') as convert_file:
        convert_file.write(json.dumps(times_dict, indent=4))

## execution tools

# get output png full name for given parameters
def get_full_name(Dataset, Algo, w_size=10, gsf=1.5, sr=10):
    filename = f"disp"
    filename += f"_{Algo}_w{w_size}_gsf{int(10*gsf)}_sr{sr}.png"
    file_full_name = os.path.join("output", Dataset, filename)
    return file_full_name, os.path.exists(file_full_name)

# get gt disparity image for a Dataset
def get_img_gt(Dataset):
    data_folder = os.path.join("output", Dataset)
    img_gt_path = os.path.join(data_folder, "disp.png")
    if not os.path.exists(img_gt_path):
        raise Exception("Image does not exist!")
    img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
    return img_gt

# run algorithm in case it wasn't run or the execution time is not benchmarked
def run_algo(Dataset, Algo, w_size=10, gsf=1.5, sr=10):
    file_full_name, _ = get_full_name(Dataset, Algo, w_size, gsf, sr)
    output_folder = os.path.join("output", Dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    times = read_times()
    time_keys = times.keys()
    if not os.path.isfile(file_full_name) or file_full_name not in time_keys:
        print(f"{file_full_name} does not exist. Computing disparity")
        command = ['./build/filters', '-HnN', f'-w{w_size}', f"-f{gsf}", f"-s{sr}", f"-m{Algo}", f"-d{Dataset}"]
        start = time.time()
        print(f"executing command {' '.join(command)}")
        process = subprocess.Popen(command)
        stdout, stderr = process.communicate()
        end = time.time()
        execution_time = end - start
        times[file_full_name] = np.round(execution_time, 2)
        write_times(times)
    return file_full_name


# comparing the results with the groundtruth using the existing metrics (run stereo if not cached)
def compare_to_gt(Dataset, Algo, w_size=10, gsf=1.5, sr=10):
    all_metrics = read_metrics()
    filename, _ = get_full_name(Dataset, Algo=Algo, w_size=w_size, gsf=gsf, sr=sr)
    # if filename in all_metrics.keys():
    #     return all_metrics[filename]

    run_algo(Dataset, Algo, w_size=w_size, gsf=gsf, sr=sr)
    img_gt = get_img_gt(Dataset)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    metrics = {}
    for metric_name, metric in metrics_dict.items():
            metrics[metric_name] = metric(img, img_gt)
    all_metrics[filename] = metrics
    write_metrics(all_metrics)
    return metrics


## define functions to get the metrics dataframes depending on a dataset

def get_metrics_Algo_func(methods):
    default_params_vals = list(default_params.values())
    def f(Dataset):
        metrics = pd.DataFrame()
        for Algo in methods:
            metrics_local = compare_to_gt(Dataset, Algo, *default_params_vals)
            for metric_name in metrics_dict.keys():
                metrics.loc[Algo, metric_name] = metrics_local[metric_name]
        return metrics
    return f

def get_metrics_param_func(Algo, param_name):
    default_params_vals = list(default_params.values())
    parameter_index = list(default_params.keys()).index(param_name)
    if(parameter_index == -1):
        ex_text = f"Parameter should be in {list(default_params.keys())}"
        raise Exception(ex_text)
    
    def f(Dataset):
        metrics = pd.DataFrame()
        for param in params[param_name]:
            default_params_vals[parameter_index] = param
            metrics_local = compare_to_gt(Dataset, Algo, *default_params_vals)
            for metric_name in metrics_dict.keys():
                metrics.loc[param, metric_name] = metrics_local[metric_name]
        return metrics
    return f


# get average metrics across Datasets for a given metric extraction function

def get_median_metrics(get_metrics_func):
    metrics_list = []
    for Dataset in datasets:
        metrics_list.append(get_metrics_func(Dataset))
    metrics_list = [i.to_xarray() for i in metrics_list]
    #getting median
    ds_all = xr.concat(metrics_list, dim='dataset')
    df_all = ds_all.to_dataframe()
    median = df_all.groupby("dataset").median()
    return median

def get_avg_metrics(get_metrics_func):
    avg_metrics = None
    for Dataset in datasets:
        if avg_metrics is None:
            avg_metrics = get_metrics_func(Dataset)
        else:
            avg_metrics += get_metrics_func(Dataset)
    # averaging
    avg_metrics = avg_metrics/len(datasets)
    return avg_metrics


# visualize image diff for a given dataset
# TODO: change
def display_image_diff(Dataset):
    f, ax = plt.subplots(1, len(methods))
    f.set_figheight(10)
    f.set_figwidth(30)
    for i, Algo in enumerate(methods):
        l = 9

        w = 1 if Algo == "DP" else 9
        img_gt = get_img_gt(Dataset)
        gt_normed = cv2.normalize(img_gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        output_folder = os.path.join("output", Algo, Dataset)
        file_full_name = run_algo(Dataset, Algo, w, l)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image = cv2.imread(file_full_name, cv2.IMREAD_GRAYSCALE)
        orig_normed = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
        image_diff = orig_normed - gt_normed
        ax[i].imshow(image_diff * 255,cmap='gray', vmin=0, vmax=255)
        ax[i].set_title(Algo)
    plt.suptitle(f"Dataset {Dataset} diff")
    plt.show()


# get execution time from the saved time json file
def get_execution_time(Dataset = None, Algo=None, params_filter= None):
    times = read_times()
    times = {key.split(".")[0]: value for key, value in times.items()}
    times = {key.split("/")[1] + "_" + "_".join(key.split("_")[1:]): value for key, value in times.items()}
    # print(times)
    if Dataset is not None: times = {key: value for key, value in times.items() if key.split("_")[0] == Dataset}
    if Algo is not None: times = {key: value for key, value in times.items() if key.split("_")[1] == Algo}

    def compare_value_and_name(param_name, param_value, key):
        words = key.split("_")
        n_params = len(default_params)
        param_encoded = [w for w in words[-n_params:] if param_name in w][0]
        param_str = param_encoded.split(param_name)[1]

        try:
            param_decoded_val = float(param_str)
        except Exception as e:
            print(param_str)
            raise e

        if param_name == "gsf":
            param_decoded_val /=10
        return abs(param_decoded_val - param_value) < 0.05

    for param_name, param_value in params_filter.items():
        p_name = params_short[param_name]
        times = {key: value for key, value in times.items() if compare_value_and_name(p_name, param_value, key)}
    return times

# get execution time for a given Dataset depending on method and window_size 
def get_time_method_param_func(param_name, methods):
    def f(Dataset):
        exec_times_ws = pd.DataFrame()
        for Algo in methods:
            for param_val in params[param_name]:
                params_filter = default_params.copy()
                params_filter[param_name] = param_val
                e_time = list(get_execution_time(Algo=Algo, Dataset=Dataset, params_filter=params_filter).values())[0]
                exec_times_ws.loc[param_val, Algo] = e_time
        return exec_times_ws
    return f

if __name__ == "__main__":

    print(compare_to_gt("Aloe", "JB", w_size=15, gsf=1.5, sr=25))

    exit(0)
    params_filter = default_params.copy()
    params_filter["gsfs"] = 0.5
    print(get_execution_time(Dataset="Books", params_filter=params_filter, Algo="JB"))
    
    exit(0)

    metrics_all = {}
    for Algo in methods:
        metrics_all[Algo] = {}
        for param in params.keys():
            metrics_all[Algo][param] = get_avg_metrics(get_metrics_param_func(Algo, param))
    metrics_all_upsample = {}
    for Algo in methods_upsample:
        metrics_all_upsample[Algo] = {}
        for param in params.keys():
            metrics_all_upsample[Algo][param] = get_avg_metrics(get_metrics_param_func(Algo, param))