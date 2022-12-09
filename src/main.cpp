#include <iostream>
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

void OurBilterelarFiler(const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
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

void JointBilterelarFiler(const cv::Mat& inputDepth, const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	// e^-r^2(x, y)/2(sigma^2)
	const auto width = input.cols;
	const auto height = input.rows;
	
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
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));
			// box filter
			int sum = 0;
			float sumBilateralMask = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(r + i, c + j ));
					int intensity_depth = static_cast<int>(inputDepth.at<uchar>(r + i, c + j));
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

void JointBileteralUpsamplingFilter(const cv::Mat& inputDepth, const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	cv::Mat depth_nearest;
	cv::resize(inputDepth, depth_nearest, input.size(), 0, 0, cv::INTER_NEAREST);
	JointBilterelarFiler(depth_nearest, input, output,window_size,sigmaSpatialFactor, sigmaRange);
	return;
}

void IterativeUpsamplingFilter(const cv::Mat& inputDepth, const cv::Mat& input, cv::Mat& output, int window_size=5, double sigmaSpatialFactor=1.0, double sigmaRange=20.0) {
	// std::cout << inputDepth.size() << std::endl;
	// std::cout << input.size() << std::endl;
	int fu = static_cast <int> (std::floor( std::log2(input.rows/inputDepth.rows) + 0.00001));

	cv::Mat D_hat =  inputDepth.clone();
	cv::Mat I_hat;
	for (size_t i = 1; i <= fu-1 ; i++)
	{
		cv::resize(D_hat, D_hat, D_hat.size()*2, 0, 0, cv::INTER_LINEAR);
		cv::resize(input, I_hat, D_hat.size(), 0, 0, cv::INTER_LINEAR);
		JointBilterelarFiler(D_hat, I_hat, D_hat, window_size, sigmaSpatialFactor, sigmaRange);
	}
	cv::resize(D_hat, D_hat, input.size(), 0, 0, cv::INTER_LINEAR);
	JointBilterelarFiler(D_hat, input, output, window_size, sigmaSpatialFactor, sigmaRange);
}


int main(int argc, char** argv) {
	fs::path default_image_path("data");
	default_image_path /= "art.png";
	fs::path default_config_path ("params.cfg");

	std::string image_path;
	std::string image_orig_path = "data/view0.png";
	std::string config_path;
	int nProcessors;
	
    	po::options_description command_line_options("cli options");
    	command_line_options.add_options()
    	("help,h", "Produce help message")
    	("version,V", "Get program version")
    	("jobs,j", po::value<int>(& nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")
    	("image,i", po::value<std::string>(& image_path)->default_value(default_image_path.string()), "Image path")
    	("config,c", po::value<std::string>(& config_path)->default_value(default_config_path.string()), "Path to the congiguration file");

	po::positional_options_description p;
    	p.add("image", 1);
	
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
	std::cout << "Filtering image " << image_path << std::endl;
	cv::Mat im = cv::imread(image_path, 0);
	cv::Mat im_orig = cv::imread(image_orig_path, 0);
	
	cv::Mat gt_copy;
	im.copyTo(gt_copy);
	gt_copy.convertTo(gt_copy, CV_16SC1);
	std::vector<std::string> filterNames;
	std::vector<double> SSIMs;
	std::vector<double> PSRNs; 

	if (im.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}

	//cv::imshow("im", im);
	//cv::waitKey();
	cv::Mat noise(im.size(), im.type());
	uchar mean = 0;
	uchar stddev = 25;
	cv::randn(noise, mean, stddev);

	im += noise;

	cv::imshow("im", im);
	// cv::waitKey();
	//std::cout << "SSIM " << my_metrics::ssim(gt_copy, im, 5) << std::endl;	
	// SSIMs.push_back(my_metrics::ssim(im, im, 5));
	// PSRNs.push_back(my_metrics::psnr(im, im, 5));
	// filterNames.push_back("Orig");
	// SSIMs.push_back(my_metrics::ssim(gt_copy, im, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, im, 5));
	// filterNames.push_back("Nois");
	// gaussian
	cv::Mat output(im.size(), CV_8U);
	// cv::GaussianBlur(im, output, cv::Size(7, 7), 0, 0);
	// std::cout << output.type() << std::endl;
	// return 0; 
	// cv::imshow("gaussian", output);
	// cv::waitKey();
	
	// SSIMs.push_back(my_metrics::ssim(gt_copy, output, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, output, 5));
	// filterNames.push_back("Gaus");

	// // median
	// cv::medianBlur(im, output, 3);
	// cv::imshow("median", output);
	// //cv::waitKey();
		
	// SSIMs.push_back(my_metrics::ssim(gt_copy, output, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, output, 5));
	// filterNames.push_back("Med");

	// // bilateral
	// double window_size = 11;
	// cv::bilateralFilter(im, output, window_size, 2 * window_size, window_size / 2);
	// cv::imshow("bilateral", output);
		
	// SSIMs.push_back(my_metrics::ssim(gt_copy, output, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, output, 5));
	// filterNames.push_back("Bilet");

	int gauss_ws = 17;
	double gauss_sigma_factor = 1.0;
	// OurGaussianFiler(im, output, gauss_ws, gauss_sigma_factor);
	// cv::imshow("OurGaussianFiler", output);

	// SSIMs.push_back(my_metrics::ssim(gt_copy, output, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, output, 5));
	// filterNames.push_back("OurGaus");

	double sigmaRange = 50.0;
	// OurBilterelarFiler(im, output, gauss_ws, gauss_sigma_factor, sigmaRange);
	// cv::imshow("OurBilterelarFiler", output);
	
	
	
	// SSIMs.push_back(my_metrics::ssim(gt_copy, output, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, output, 5));
	// filterNames.push_back("OurBilet");

	// JointBilterelarFiler(im, im_orig, output, gauss_ws, gauss_sigma_factor, sigmaRange);
	// cv::imshow("JointBilterelarFiler", output);

	// cv::waitKey();

  	cv::Mat low_res;
	cv::resize(im, low_res, im.size()/2, cv::INTER_LINEAR);
	// cv::imshow("ResizedDown", low_res);

	// cv::waitKey();
	// cv::Mat output_low_res(low_res.size(), CV_8U);
	// im_orig_low_res;

	JointBileteralUpsamplingFilter(low_res, im_orig, output, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imshow("JBU", output);

	IterativeUpsamplingFilter(low_res, im_orig, output, gauss_ws, gauss_sigma_factor, sigmaRange);
	cv::imshow("Iterative", output);

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
