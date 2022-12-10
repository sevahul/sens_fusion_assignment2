#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options/options_description.hpp>
#include <omp.h>
#include <cmath>
#include "metrics.hpp"
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ch = std::chrono;

const std::string version = "1.0";

void GeneralFiler(const cv::Mat& input, cv::Mat& output, cv::Mat mask) {
	std::cout << "applying the following filter:" << std::endl;
	std::cout << mask << std::endl;
	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = mask.cols;
	const int hw = window_size/2;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output.at<uchar>(i, j) = 0;
		}	
	}
	
	// normalize mask
	int sum_mask = 0;
	for (int i = 0; i < window_size; i++) {
		for (int j = 0; j < window_size; j ++) {
			sum_mask += static_cast<int>(mask.at<uchar>(i, j));
		}
	}
	

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			// box filter
			int sum = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j ));
					int weight =  static_cast<int>(mask.at<uchar>(i + window_size/2, j + window_size/2));
					sum += intensity * weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_mask;
		}
	}
}
void CreateGaussianMask(int window_size, cv::Mat& mask, double sigma)
{
	
	const double sigmaSq = sigma * sigma;
	mask = cv::Mat(window_size, window_size, CV_8UC1);
	
	const int hw = window_size/2;
	for (int i = 0; i < window_size; ++i) {
		for (int j = 0; j < window_size; ++j) {
			double r2 = (i - hw) * (i - hw) + (j - hw) * (j - hw);
			mask.at<uchar>(i, j) = 255 * std::exp(-r2/(2*sigmaSq));
			
		}
	}
}

void OurGaussianFiler(const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaFactor=1.0) {
	// e^-r^2(x, y)/2(sigma^2)
	const auto width = input.cols;
	const auto height = input.rows;
	
	const int hw = window_size/2;
	const double sigma = hw/2.5 * sigmaFactor;

	cv::Mat mask;
	CreateGaussianMask(window_size, mask, sigma);	
	// normalize mask
	GeneralFiler(input, output, mask);
}

void BilterelarFiler(const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	// e^-r^2(x, y)/2(sigma^2)
	const auto width = input.cols;
	const auto height = input.rows;
	
	const int hw = window_size/2;
	const double sigmaSpatial = hw/2.5 * sigmaSpatialFactor;

	cv::Mat mask;
	CreateGaussianMask(window_size, mask, sigmaSpatial);	

	const float sigmaRangeSq = sigmaRange * sigmaRange;	

	float range_mask[256];


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output.at<uchar>(i, j) = 0;
		}	
	}

	// compute kernel range
	for (int diff = 0; diff < 256; diff++) {
		range_mask[diff] = std::exp(- diff * diff / (2*sigmaRangeSq));
	}

	// normalize mask
	
	

	int sum_mask = 0;
	for (int i = 0; i < window_size; i++) {
		for (int j = 0; j < window_size; j ++) {
			sum_mask += static_cast<int>(mask.at<uchar>(i, j));
		}
	}
	

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));
			// box filter
			int sum = 0;
			float sumBilateralMask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j ));
					int diff = std::abs(intensity_center - intensity);
					float weight_range =  range_mask[diff];
					int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size/2, j + window_size/2));
					float weight = weight_range * weight_spatial;
					sum += intensity * weight;
					sumBilateralMask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sumBilateralMask;
		}
	}
}

void JointBilterelarFiler(const cv::Mat& D, const cv::Mat& I, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	// e^-r^2(x, y)/2(sigma^2)
	const auto width = I.cols;
	const auto height = I.rows;
	
	const int hw = window_size/2;
	const double sigmaSpatial = hw/2.5 * sigmaSpatialFactor;

	cv::Mat mask;
	CreateGaussianMask(window_size, mask, sigmaSpatial);
	
	cv::Mat output_copy = output.clone();	

	const float sigmaRangeSq = sigmaRange * sigmaRange;	

	float range_mask[256];
	


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output_copy.at<uchar>(i, j) = 0;
		}	
	}

	// compute kernel range
	for (int diff = 0; diff < 256; diff++) {
		range_mask[diff] = std::exp(- diff * diff / (2*sigmaRangeSq));
	}

	// normalize mask
	int sum_mask = 0;
	for (int i = 0; i < window_size; i++) {
		for (int j = 0; j < window_size; j ++) {
			sum_mask += static_cast<int>(mask.at<uchar>(i, j));
		}
	}
	
	#pragma omp parallel for
	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			int intensity_center = static_cast<int>(I.at<uchar>(r, c));
			// box filter
			int sum = 0;
			float sumBilateralMask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(I.at<uchar>(r + i, c + j ));
					int intensity_depth = static_cast<int>(D.at<uchar>(r + i, c + j));
					int diff = std::abs(intensity_center - intensity);
					float weight_range =  range_mask[diff];
					int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size/2, j + window_size/2));
					float weight = weight_range * weight_spatial;
					sum += intensity_depth * weight;
					sumBilateralMask += weight;
				}
			}
			output_copy.at<uchar>(r, c) = sum / sumBilateralMask;
		}
	}
	output = output_copy.clone();
}

void JointBileteralUpsamplingFilter(const cv::Mat& D, const cv::Mat& I, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	cv::Mat D_upsampled;
	cv::resize(D, D_upsampled, I.size(), 0, 0, cv::INTER_NEAREST);
	JointBilterelarFiler(D_upsampled, I, output,window_size,sigmaSpatialFactor, sigmaRange);
	return;
}

void IterativeUpsamplingFilter(const cv::Mat& D, const cv::Mat& I, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	int fu = static_cast <int> (std::floor( std::log2(I.rows/D.rows) + 0.00001));

	cv::Mat D_hat =  D.clone();
	cv::Mat I_hat;
	for (size_t i = 1; i <= fu-1 ; i++)
	{
		cv::resize(D_hat, D_hat, D_hat.size()*2, 0, 0, cv::INTER_LINEAR);
		cv::resize(I, I_hat, D_hat.size(), 0, 0, cv::INTER_LINEAR);
		JointBilterelarFiler(D_hat, I_hat, D_hat, window_size, sigmaSpatialFactor, sigmaRange);
	}
	cv::resize(D_hat, D_hat, I.size(), 0, 0, cv::INTER_LINEAR);
	JointBilterelarFiler(D_hat, I, output, window_size, sigmaSpatialFactor, sigmaRange);
}

void readDataset(const std::string& datasetName, cv::Mat& I_out, cv::Mat& D_out_gt, cv::Mat& D_out, cv::Mat& D_out_downsampled, int& dmin) {
	fs::path data_path ("data");
	data_path = data_path/datasetName;

	fs::path I_path = data_path/"intensity.png";
	fs::path D_path = data_path/"disp_DP.png";
	fs::path D_gt_path = data_path/"disparity.png";
	fs::path dmin_path = data_path/"dmin.txt";
	fs::path output_dir ("output");
	output_dir /= datasetName;

	I_out = cv::imread(I_path.string(), 0);
	if (I_out.data == nullptr) std::cerr << "Failed to load image intensity image for " << datasetName << std::endl;
	D_out_gt = cv::imread(D_gt_path.string(), 0);
	if (D_out_gt.data == nullptr) std::cerr << "Failed to load groundtruth disparity image for " << datasetName << std::endl;
	D_out = cv::imread(D_path.string(), 0);
	if (D_out.data == nullptr) std::cerr << "Failed to load image disparity image for " << datasetName << std::endl;

	std::ifstream MyReadFile(dmin_path.string());
	std::string dmin_string;
	std::getline (MyReadFile, dmin_string);
	dmin = std::stoi(dmin_string);
	MyReadFile.close();

	cv::resize(D_out, D_out_downsampled, D_out_gt.size()/4, 0, 0, cv::INTER_LINEAR);
	// fs::create_directory("output");
	fs::create_directories(output_dir);

}

void Disparity2PointCloud(
    const std::string& output_file, cv::Mat& disparities,
    const int& dmin, const double& baseline = 160, const double& focal_length = 3740);

int main(int argc, char** argv) {
	fs::path default_config_path ("params.cfg");
	std::string defaultDatasetName = "Art";
	std::string datasetName;
	bool show_images = true;
	bool gen_pointclouds = true;
	
	
	std::string config_path;
	int nProcessors;
	const double focal_length = 3740;
	const double baseline = 160;
	int dmin;
	std::string method;
	
    po::options_description command_line_options("cli options");
    command_line_options.add_options()
    	("help,h", "Produce help message")
    	("version,V", "Get program version")
    	("jobs,j", po::value<int>(& nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")
    	("dataset,d", po::value<std::string>(& datasetName)->default_value(defaultDatasetName), "Dataset Name")
    	("config,c", po::value<std::string>(& config_path)->default_value(default_config_path.string()), "Path to the congiguration file");
    	("method,m", po::value<std::string>(& method)->default_value("all"), "method name: all/Bilet/JB/JBU/Iter");

	po::positional_options_description p;
    	p.add("dataset", 1);
	
	po::variables_map vm;
    	po::options_description cmd_opts;
    	cmd_opts.add(command_line_options);
    	po::store(po::command_line_parser(argc, argv).
        options(cmd_opts).positional(p).run(), vm);
    	po::notify(vm);
    
    	//po::store(po::command_line_parser(0, 0).options(config_only_options).run(), vm);
    	notify(vm);

	
	if (vm.count("help")) {
		std::cout << "Usage: OpenCV_stereo [<left-image> [<right-image> [<output>]]] [<options>]\n";
		po::options_description help_opts;
		help_opts.add(command_line_options);
		std::cout << help_opts << "\n";
		return 1;
	}

	if (vm.count("version")) {
		std::cout << "Image filtering " << version << std::endl;
		return 1;
	}
	std::cout << "Processing dataset " << datasetName << std::endl;
	fs::path output_dir ("output");
	output_dir /= datasetName;

	cv::Mat I;
	cv::Mat D_gt;
	cv::Mat D_dp;
	cv::Mat D_ds;

	readDataset(datasetName, I, D_gt, D_dp, D_ds, dmin);
	
	cv::imwrite((output_dir/"disp.png").string(), D_gt);
	cv::imwrite((output_dir/"disp_DP.png").string(), D_dp);
	
	
	// cv::Mat I_noisy;
	// I.copyTo(I_noisy);
	// cv::Mat noise(I_noisy.size(), I_noisy.type());
	// uchar mean = 0;
	// uchar stddev = 25;
	// cv::randn(noise, mean, stddev);
	// I_noisy += noise;

	// I_noisy.convertTo(I_noisy, CV_16SC1);
	// std::vector<std::string> filterNames;
	// std::vector<double> SSIMs;
	// std::vector<double> PSRNs; 

	if (D_gt.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}

	if (show_images) cv::imshow("D_gt", D_gt);
	

	if(show_images) cv::imshow("I", I);
	// cv::waitKey();
	//std::cout << "SSIM " << my_metrics::ssim(I_noisy, im, 5) << std::endl;	
	// SSIMs.push_back(my_metrics::ssim(im, im, 5));
	// PSRNs.push_back(my_metrics::psnr(im, im, 5));
	// filterNames.push_back("Orig");
	// SSIMs.push_back(my_metrics::ssim(I_noisy, im, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, im, 5));
	// filterNames.push_back("Nois");
	// gaussian
	cv::Mat output_I(I.size(), CV_8U);
	cv::Mat output_D(I.size(), CV_8U);

	int gauss_ws = 10;
	double gauss_sigma_factor = 1;

	double sigmaRange = 10.0;
	BilterelarFiler(D_dp, output_D, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imwrite((output_dir/"disp_Bilet.png").string(), output_D);
	if (gen_pointclouds) Disparity2PointCloud((output_dir/"pcl_Bilet").string(), output_D, dmin);

	if(show_images) cv::imshow("Bilterelar Filer D", output_D);

	JointBilterelarFiler(D_dp, I, output_D, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imwrite((output_dir/"disp_JB.png").string(), output_D);
	if (gen_pointclouds) Disparity2PointCloud((output_dir/"pcl_JB").string(), output_D, dmin);
	if(show_images) cv::imshow("Joint Bilterelar Filer D", output_D);

	JointBileteralUpsamplingFilter(D_ds, I, output_D, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imwrite((output_dir/"disp_JBU.png").string(), output_D);
	if (gen_pointclouds) Disparity2PointCloud((output_dir/"pcl_JBU").string(), output_D, dmin);
	if(show_images) cv::imshow("JBU D", output_D);

	IterativeUpsamplingFilter(D_ds, I, output_D, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imwrite((output_dir/"disp_iter.png").string(), output_D);
	if (gen_pointclouds) Disparity2PointCloud((output_dir/"pcl_iter").string(), output_D, dmin);
	if(show_images) cv::imshow("Iterative Updsampling D", output_D);

	// cv::waitKey();
	
	// std::cout << "Filts: "; 	
	// for (std::string i: filterNames)
    // 		std::cout << i << '\t';
	// std::cout << std::endl;
	
	// std::cout << "SSIMs: "; 	
	// for (double i: SSIMs)
    // 		std::cout << ceil(i * 100.0) / 100.0 << '\t';
	// std::cout << std::endl;
	// std::cout << "PSNRs: "; 	
	// for (double i: PSRNs)
    // 		std::cout << ceil(i * 100.0) / 100.0 << '\t';
	// std::cout << std::endl; 

	cv::waitKey();
	return 0;
}

void Disparity2PointCloud(
    const std::string& output_file, cv::Mat& disparities,
    const int& dmin, const double& baseline, const double& focal_length)
{
	int rows = disparities.rows;
	int cols = disparities.cols;
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());

    for (int r = 0; r < rows; ++r) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((r) / static_cast<double>(rows + 1)) * 100) << "%\r" << std::flush;
		// #pragma omp parallel for
		for (int c = 0; c < cols; ++c) {
            if (disparities.at<uchar>(r, c) == 0) continue;

            int d = (int)disparities.at<uchar>(r, c) + dmin;
            int u1 = c - cols/2;
            int u2 = c + d - cols/2;
	    	int v1 = r - rows/2;

            const double Z = baseline * focal_length / d;
            const double X = -0.5 * ( baseline * (u1 + u2) ) / d;
            const double Y = baseline * v1 / d;
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}