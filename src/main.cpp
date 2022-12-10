#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options/options_description.hpp>
#include <omp.h>
#include <cmath>
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui_c.h>
#include "metrics.hpp"
#include "filters.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ch = std::chrono;

const std::string version = "1.0";

int gaussWs = 10, gaussWsPrev = gaussWs;
double gaussSigmaFactor = 1, gaussSigmaFactorPrev = gaussSigmaFactor;
double sigmaRange = 10, sigmaRangePrev = sigmaRange;
bool changed = true;

void funcGaussWs(int newValue, void *object)
{
	gaussWs = newValue;
	changed = true;
};

void funcGaussSigmaFactor(int newValue, void *object)
{
	gaussSigmaFactor = ((double)newValue)/10;
	changed = true;
};

void funcSigmaRange(int newValue, void *object)
{
	sigmaRange = newValue;
};

int main(int argc, char **argv)
{
	fs::path default_config_path("params.cfg");
	std::string defaultDatasetName = "Art";
	std::string datasetName;
	bool show_images = false;
	bool gen_pointclouds = false;

	std::string config_path;
	int nProcessors;
	const double focal_length = 3740;
	const double baseline = 160;
	int dmin;
	std::string method;

	po::options_description command_line_options("cli options");
	command_line_options.add_options()("help,h", "Produce help message")("version,V", "Get program version")("jobs,j", po::value<int>(&nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")("dataset,d", po::value<std::string>(&datasetName)->default_value(defaultDatasetName), "Dataset Name")("config,c", po::value<std::string>(&config_path)->default_value(default_config_path.string()), "Path to the congiguration file");
	("method,m", po::value<std::string>(&method)->default_value("all"), "method name: all/Bilet/JB/JBU/Iter");

	po::positional_options_description p;
	p.add("dataset", 1);

	po::variables_map vm;
	po::options_description cmd_opts;
	cmd_opts.add(command_line_options);
	po::store(po::command_line_parser(argc, argv).options(cmd_opts).positional(p).run(), vm);
	po::notify(vm);

	// po::store(po::command_line_parser(0, 0).options(config_only_options).run(), vm);
	notify(vm);

	if (vm.count("help"))
	{
		std::cout << "Usage: OpenCV_stereo [<left-image> [<right-image> [<output>]]] [<options>]\n";
		po::options_description help_opts;
		help_opts.add(command_line_options);
		std::cout << help_opts << "\n";
		return 1;
	}

	if (vm.count("version"))
	{
		std::cout << "Image filtering " << version << std::endl;
		return 1;
	}
	std::cout << "Processing dataset " << datasetName << std::endl;
	fs::path output_dir("output");
	output_dir /= datasetName;

	cv::Mat I;
	cv::Mat D_gt;
	cv::Mat D_dp;
	cv::Mat D_ds;

	my_filters::readDataset(datasetName, I, D_gt, D_dp, D_ds, dmin);

	cv::imwrite((output_dir / "disp.png").string(), D_gt);
	cv::imwrite((output_dir / "disp_DP.png").string(), D_dp);

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

	if (D_gt.data == nullptr)
	{
		std::cerr << "Failed to load image" << std::endl;
	}

	if (show_images)
		cv::imshow("D_gt", D_gt);

	if (show_images)
		cv::imshow("I", I);
	// cv::waitKey();
	// std::cout << "SSIM " << my_metrics::ssim(I_noisy, im, 5) << std::endl;
	// SSIMs.push_back(my_metrics::ssim(im, im, 5));
	// PSRNs.push_back(my_metrics::psnr(im, im, 5));
	// filterNames.push_back("Orig");
	// SSIMs.push_back(my_metrics::ssim(I_noisy, im, 5));
	// PSRNs.push_back(my_metrics::psnr(gt_copy, im, 5));
	// filterNames.push_back("Nois");
	// gaussian
	cv::Mat output_I(I.size(), CV_8U);
	cv::Mat output_D_bilet(I.size(), CV_8U);
	cv::Mat output_D_JB(I.size(), CV_8U);
	cv::Mat output_D_JBU(I.size(), CV_8U);
	cv::Mat output_D_Iter(I.size(), CV_8U);

	my_filters::BilterelarFiler(D_dp, output_D_bilet, gaussWs, gaussSigmaFactor, sigmaRange);
	cv::imwrite((output_dir / "disp_Bilet.png").string(), output_D_bilet);
	if (gen_pointclouds)
		my_filters::Disparity2PointCloud((output_dir / "pcl_Bilet").string(), output_D_bilet, dmin);
	if (show_images)
		cv::imshow("Bilterelar Filer D", output_D_bilet);

	my_filters::JointBilterelarFiler(D_dp, I, output_D_JB, gaussWs, gaussSigmaFactor, sigmaRange);
	cv::imwrite((output_dir / "disp_JB.png").string(), output_D_JB);
	if (gen_pointclouds)
		my_filters::Disparity2PointCloud((output_dir / "pcl_JB").string(), output_D_JB, dmin);
	if (show_images)
		cv::imshow("Joint Bilterelar Filer D", output_D_JB);

	my_filters::JointBileteralUpsamplingFilter(D_ds, I, output_D_JBU, gaussWs, gaussSigmaFactor, sigmaRange);
	cv::imwrite((output_dir / "disp_JBU.png").string(), output_D_JBU);
	if (gen_pointclouds)
		my_filters::Disparity2PointCloud((output_dir / "pcl_JBU").string(), output_D_JBU, dmin);
	if (show_images)
		cv::imshow("JBU D", output_D_JBU);

	my_filters::IterativeUpsamplingFilter(D_ds, I, output_D_Iter, gaussWs, gaussSigmaFactor, sigmaRange);
	cv::imwrite((output_dir / "disp_iter.png").string(), output_D_Iter);
	if (gen_pointclouds)
		my_filters::Disparity2PointCloud((output_dir / "pcl_iter").string(), output_D_Iter, dmin);
	if (show_images)
		cv::imshow("Iterative Updsampling D", output_D_Iter);

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

	if (show_images)
		cv::waitKey();
	

	bool val_changed = true;
	bool interact = true;
	if (interact)
	{
		// Declaration of the names of the windows
		std::string nameBilet = "imgBilet";
		std::string nameJB = "imgJB";
		std::string nameJBU = "imgJBU";
		std::string nameIter = "imgIter";

		// Creating the windows
		cv::namedWindow(nameBilet, cv::WINDOW_AUTOSIZE);
		cv::namedWindow(nameJB, cv::WINDOW_AUTOSIZE);
		cv::namedWindow(nameJBU, cv::WINDOW_AUTOSIZE);
		cv::namedWindow(nameIter, cv::WINDOW_AUTOSIZE);
		cv::createTrackbar("GaussSigmaFactor", nameJB, NULL, 100, funcGaussSigmaFactor);
		cv::setTrackbarPos("GaussSigmaFactor", nameJB, int(gaussSigmaFactor)*10 );
		cv::createTrackbar("GaussWs", nameJB, NULL, 100, funcGaussWs);
		cv::setTrackbarPos("GaussWs", nameJB, gaussWs);
		// cv::setTrackbarPos("GaussWs", nameJB, NULL, 100, funcGaussWs);
		// cv::createTrackbar("SigmaRange", nameJB, NULL, 100, funcSigmaRange);

		bool closedBilet = false;
		bool closedJB = false;
		bool closedJBU = false;
		bool closedIter = false;
		bool visiable = true;
		char charCheckForESCKey{0};
		while (charCheckForESCKey != 27 && visiable)
		{ // loop until ESC key is pressed or webcam is lost
			// bool frameSuccess = webCam.read(imgOriginal); // get next frame from input stream

			// if (!frameSuccess || imgOriginal.empty()){ // if the frame wasnot read or read wrongly
			// 	std::cerr << "error: Frame could not be read." << std::endl;
			// 	break;
			// }

			// cv::cvtColor(imgOriginal, imgGrayScale, cv::COLOR_BGR2GRAY); // original video is converted to grayscale into imgGrayScale

			// cv::GaussianBlur(imgGrayScale, imgBlurred, cv::Size(5, 5), 1.8); // blurrs the grayscale video. Check OpenCV docs for explanation of parameters
			// cv::Canny(imgBlurred, imgCanny, lowTh, highTh); // Canny edge detection. Check OpenCV docs for explanation of parameters
			// cv::threshold(imgGrayScale, imgThreshold, th, maxVal, cv::THRESH_BINARY);
			// cv::adaptiveThreshold(imgGrayScale, imgAdaptiveThreshold, maxVal,
			// 							cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
			// 									blockSize*2 + 3, C);
			// Declaration of windows for output video results. Check OpenCV docs for explanation of parameters

			// Show output video results windows
			// If the window is not closed
			if (changed) {
				if (!closedBilet)
				{
					my_filters::JointBilterelarFiler(D_dp, I, output_D_JB, gaussWs, gaussSigmaFactor, sigmaRange);
					cv::imshow(nameBilet, output_D_bilet);
				}
				if (!closedJB)
				{
					cv::imshow(nameJB, output_D_JB);
				}
				if (!closedJBU)
				{
					cv::imshow(nameJBU, output_D_JBU);
				}
				if (!closedIter)
				{
					cv::imshow(nameIter, output_D_Iter);
				}
				changed = false;
			}

			charCheckForESCKey = cv::waitKey(1); // gets the key pressed

			// check which windows are closed
			closedBilet = (cvGetWindowHandle(nameBilet.c_str()) == 0);
			closedJB = (cvGetWindowHandle(nameJB.c_str()) == 0);
			closedJBU = (cvGetWindowHandle(nameJBU.c_str()) == 0);
			closedIter = (cvGetWindowHandle(nameIter.c_str()) == 0);

			// If all windows are closed, end the program
			if (closedBilet && closedJB && closedJBU && closedIter)
			{
				visiable = false;
			}
		}
	}

	return 0;
}