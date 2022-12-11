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


// TODO
// - param for meaningful names

const std::string version = "1.0";

// Declaration of the names of the windows
std::string nameBilet = "imgBilet";
std::string nameJB = "imgJB";
std::string nameJBU = "imgJBU";
std::string nameIter = "imgIter";

int w_size = 10;
double gaussSigmaFactor = 1;
int sigmaRange = 10;
bool changed = true;
bool changedSlider = false;

cv::Rect button;
std::string buttonText("Apply");
cv::Mat canvas;
void callButton(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (button.contains(cv::Point(x, y)))
        {
            std::cout << "Clicked!" << std::endl;
			changed = true;
        }
    }
    cv::waitKey(1);
}

void funcWs(int newValue, void *object)
{
	w_size = newValue;
	changedSlider = true;
};

void funcGaussSigmaFactor(int newValue, void *object)
{
	gaussSigmaFactor = ((double)newValue)/10.0;
	changedSlider = true;
};

void funcSigmaRange(int newValue, void *object)
{
	sigmaRange = newValue;
	changedSlider = true;
};

std::string produceName(std::string name, bool meaningful, int w_size, double gaussSigmaFactor, int sigmaRange)
{
	std::string result = name;
	if (meaningful){
		std::ostringstream ss;
		ss << std::fixed << std::setprecision(2) << name << "_w" << w_size << "_gsf" 
							<< int(gaussSigmaFactor*10) << "_sr" << sigmaRange;
		result = ss.str();
	}
	return result + ".png";
};

int main(int argc, char **argv)
{
	fs::path default_config_path("params.cfg");
	std::string defaultDatasetName = "Art";
	std::string datasetName;
	bool show_images = true;
	bool gen_pointclouds = false;
	bool interact = true;
	bool meaningfulNaming = false;

	std::string config_path;
	int nProcessors;
	const double focal_length = 3740;
	const double baseline = 160;
	int dmin;
	std::string method;

	po::options_description command_line_options("cli options");
	command_line_options.add_options()
	("help,h", "Produce help message")("version,V", "Get program version")
	("hide-disp,H", "Hide disparity images (does not affect gui)")
	("no-gui,n", "Hide gui")
	("meaningful-naming,N", "Meaningful naming")
	("gen-pcl,p", "Generate pointcloud")
	("jobs,j", po::value<int>(&nProcessors)->default_value(omp_get_max_threads()), "Number of Threads (max by default)")
	("dataset,d", po::value<std::string>(&datasetName)->default_value(defaultDatasetName), "Dataset Name")
	("config,c", po::value<std::string>(&config_path)->default_value(default_config_path.string()), "Path to the congiguration file")
	("method,m", po::value<std::string>(&method)->default_value("all"), "method name: all/Bilet/JB/JBU/Iter")
	("gauss-sigma-factor,f", po::value<double>(&gaussSigmaFactor)->default_value(gaussSigmaFactor), "Gauss sigma factor (it would adjust on the window size)")
	("window-size,w", po::value<int>(&w_size)->default_value(w_size), "Window size")
	("range-sigma,s", po::value<int>(&sigmaRange)->default_value(sigmaRange), "Sigma for range");


	po::positional_options_description p;
	p.add("dataset", 1);

	po::variables_map vm;
	po::options_description cmd_opts;
	cmd_opts.add(command_line_options);
	po::store(po::command_line_parser(argc, argv).options(cmd_opts).positional(p).run(), vm);
	po::notify(vm);

	notify(vm);
	if (vm.count("no-gui")){
		interact = false;
	}

	if (vm.count("hide-disp")){
		show_images = false;
	}
	if (vm.count("meaningful-naming")){
		meaningfulNaming = true;
	}
	if (vm.count("gen-pcl")){
		gen_pointclouds = true;
	}

	if (vm.count("help"))
	{
		std::cout << "Usage: [Dataset Name] [<options>]\n";
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

	if (D_gt.data == nullptr)
	{
		std::cerr << "Failed to load image" << std::endl;
	}

	if (show_images)
		cv::imshow("D_gt", D_gt);

	if (show_images)
		cv::imshow("I", I);

	cv::Mat output_I(I.size(), CV_8U);
	cv::Mat output_D_bilet(I.size(), CV_8U);
	cv::Mat output_D_JB(I.size(), CV_8U);
	cv::Mat output_D_JBU(I.size(), CV_8U);
	cv::Mat output_D_Iter(I.size(), CV_8U);
	std::string filename;

	if ((method == "all") || (method == "Bilet")){
		my_filters::BilterelarFiler(D_dp, output_D_bilet, w_size, gaussSigmaFactor, sigmaRange);
		filename = produceName((output_dir / "disp_Bilet").string(), meaningfulNaming, w_size, gaussSigmaFactor, sigmaRange);
		cv::imwrite(filename, output_D_bilet);
		if (gen_pointclouds)
			my_filters::Disparity2PointCloud((output_dir / "pcl_Bilet").string(), output_D_bilet, dmin);
		if (show_images)
			cv::imshow("Bilterelar Filer D", output_D_bilet);
	}

	if ((method == "all") || (method == "JB")){
		my_filters::JointBilterelarFiler(D_dp, I, output_D_JB, w_size, gaussSigmaFactor, sigmaRange);
		filename = produceName((output_dir / "disp_JB").string(), meaningfulNaming, w_size, gaussSigmaFactor, sigmaRange);
		cv::imwrite(filename, output_D_JB);
		if (gen_pointclouds)
			my_filters::Disparity2PointCloud((output_dir / "pcl_JB").string(), output_D_JB, dmin);
		if (show_images)
			cv::imshow("Joint Bilterelar Filer D", output_D_JB);
	}

	if ((method == "all") || (method == "JBU")){
		my_filters::JointBileteralUpsamplingFilter(D_ds, I, output_D_JBU, w_size, gaussSigmaFactor, sigmaRange);
		filename = produceName((output_dir / "disp_JBU").string(), meaningfulNaming, w_size, gaussSigmaFactor, sigmaRange);
		cv::imwrite(filename, output_D_JBU);
		if (gen_pointclouds)
			my_filters::Disparity2PointCloud((output_dir / "pcl_JBU").string(), output_D_JBU, dmin);
		if (show_images)
			cv::imshow("JBU D", output_D_JBU);
	}

	if ((method == "all") || (method == "Iter")){
		my_filters::IterativeUpsamplingFilter(D_ds, I, output_D_Iter, w_size, gaussSigmaFactor, sigmaRange);
		filename = produceName((output_dir / "disp_Iter").string(), meaningfulNaming, w_size, gaussSigmaFactor, sigmaRange);
		cv::imwrite(filename, output_D_Iter);
		if (gen_pointclouds)
			my_filters::Disparity2PointCloud((output_dir / "pcl_iter").string(), output_D_Iter, dmin);
		if (show_images)
			cv::imshow("Iterative Updsampling D", output_D_Iter);
	}


	if (show_images)
		cv::waitKey();
	
	if (interact)
	{
		// Creating the windows
		button = cv::Rect(0,0,I.cols, 50);
		canvas = cv::Mat(I.rows + button.height, I.cols, 0);
		canvas(button) = 200;
		putText(canvas(button), buttonText, cv::Point(button.width*0.35, button.height*0.7), cv::FONT_HERSHEY_PLAIN, 1, 0);

		std::vector<std::string> windowNames;
		if ((method == "all") || (method == "Bilet")) windowNames.push_back(nameBilet);
		if ((method == "all") || (method == "JB")) windowNames.push_back(nameJB);
		if ((method == "all") || (method == "JBU")) windowNames.push_back(nameJBU);
		if ((method == "all") || (method == "Iter")) windowNames.push_back(nameIter);

		for(std::string wName : windowNames)
  			cv::namedWindow(wName, cv::WINDOW_AUTOSIZE);
		for(std::string wName : windowNames)
			cv::setMouseCallback(wName, callButton);
		for(std::string wName : windowNames){
			cv::createTrackbar("GaussSigmaFactor*10", wName, NULL, 100, funcGaussSigmaFactor);
			cv::setTrackbarPos("GaussSigmaFactor*10", wName, int(gaussSigmaFactor*10) );
			cv::createTrackbar("WindowSize", wName, NULL, 100, funcWs);
			cv::setTrackbarPos("WindowSize", wName, w_size);
			cv::createTrackbar("RangeSigma", wName, NULL, 100, funcSigmaRange);
			cv::setTrackbarPos("RangeSigma", wName, sigmaRange);
		}
		// cv::setTrackbarPos("WindowSize", nameJB, NULL, 100, funcWs);
		// cv::createTrackbar("SigmaRange", nameJB, NULL, 100, funcSigmaRange);

		bool closedBilet = false || !((method == "all") || (method == "Bilet"));
		bool closedJB = false || !((method == "all") || (method == "JB"));
		bool closedJBU = false || !((method == "all") || (method == "JBU"));
		bool closedIter = false || !((method == "all") || (method == "Iter"));

		bool visiable = true;
		char charCheckForESCKey{0};
		while (charCheckForESCKey != 27 && visiable)
		{ 
			// Show output video results windows
			// If the window is not closed
			if (changedSlider){
				for(std::string wName : windowNames) {
					cv::setTrackbarPos("GaussSigmaFactor*10", wName.c_str(), int(gaussSigmaFactor*10));
					cv::setTrackbarPos("WindowSize", wName.c_str(), w_size);
					cv::setTrackbarPos("RangeSigma", wName.c_str(), sigmaRange);
				}
			}
			double first_ssim = my_metrics::ssim(D_dp, D_gt, 5);
			std::string DP_ssim_str = " DP ssim: " + std::to_string(first_ssim);
			if (changed) {
				// std::cout << "Changed" << std::endl;

				std::ostringstream ss;
				ss << std::fixed << std::setprecision(2) << "Window Size: " <<  w_size <<
									"; Gauss Sigma Factor: " << gaussSigmaFactor << 
									"; Range Sigma: " << sigmaRange;
				std::string text = ss.str();

				
				if (!closedBilet)
				{
					std::cout << "applying"  << std::endl;
					my_filters::BilterelarFiler(D_dp, output_D_bilet, w_size, gaussSigmaFactor, sigmaRange);
					double ssim = my_metrics::ssim(output_D_bilet, D_gt, 5);
					cv::putText(output_D_bilet, text, cv::Point(10,30), 1, 1, 255, 1);
					cv::putText(output_D_bilet, "Cur ssim: " + std::to_string(ssim), cv::Point(10,50), 1, 1, 255, 1);
					cv::putText(output_D_bilet, DP_ssim_str, cv::Point(10,70), 1, 1, 255, 1);
					output_D_bilet.copyTo(canvas(cv::Rect(0, button.height, I.cols, I.rows)));
					cv::imshow(nameBilet, canvas);
				}
				if (!closedJB)
				{
					std::cout << "applying"  << std::endl;
					my_filters::JointBilterelarFiler(D_dp, I, output_D_JB, w_size, gaussSigmaFactor, sigmaRange);
					double ssim = my_metrics::ssim(output_D_JB, D_gt, 5);
					cv::putText(output_D_JB, text, cv::Point(10,30), 1, 1, 255, 1);
					cv::putText(output_D_JB, "Cur ssim: " + std::to_string(ssim), cv::Point(10,50), 1, 1, 255, 1);
					cv::putText(output_D_JB, DP_ssim_str, cv::Point(10,70), 1, 1, 255, 1);
					output_D_JB.copyTo(canvas(cv::Rect(0, button.height, I.cols, I.rows)));
					cv::imshow(nameJB, canvas);
				}
				if (!closedJBU)
				{
					my_filters::JointBileteralUpsamplingFilter(D_ds, I, output_D_JBU, w_size, gaussSigmaFactor, sigmaRange);
					double ssim = my_metrics::ssim(output_D_JBU, D_gt, 5);
					cv::putText(output_D_JBU, text, cv::Point(10,30), 1, 1, 255, 1);
					cv::putText(output_D_JBU, "Cur ssim: " + std::to_string(ssim), cv::Point(10,50), 1, 1, 255, 1);
					cv::putText(output_D_JBU, DP_ssim_str, cv::Point(10,70), 1, 1, 255, 1);
					output_D_JBU.copyTo(canvas(cv::Rect(0, button.height, I.cols, I.rows)));
					cv::imshow(nameJBU, canvas);
				}
				if (!closedIter)
				{
					std::cout << "applying"  << std::endl;
					my_filters::IterativeUpsamplingFilter(D_ds, I, output_D_Iter, w_size, gaussSigmaFactor, sigmaRange);
					double ssim = my_metrics::ssim(output_D_Iter, D_gt, 5);
					cv::putText(output_D_Iter, text, cv::Point(10,30), 1, 1, 255, 1);
					cv::putText(output_D_Iter, "Cur ssim: " + std::to_string(ssim), cv::Point(10,50), 1, 1, 255, 1);
					cv::putText(output_D_Iter, DP_ssim_str, cv::Point(10,70), 1, 1, 255, 1);
					output_D_Iter.copyTo(canvas(cv::Rect(0, button.height, I.cols, I.rows)));
					cv::imshow(nameIter, canvas);
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