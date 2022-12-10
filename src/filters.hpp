#ifndef MY_FILTERS
#define MY_FILTERS

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
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
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace ch = std::chrono;

namespace my_filters
{
    void GeneralFiler(const cv::Mat &input, cv::Mat &output, cv::Mat mask)
    {
        std::cout << "applying the following filter:" << std::endl;
        std::cout << mask << std::endl;
        const auto width = input.cols;
        const auto height = input.rows;

        const int window_size = mask.cols;
        const int hw = window_size / 2;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                output.at<uchar>(i, j) = 0;
            }
        }

        // normalize mask
        int sum_mask = 0;
        for (int i = 0; i < window_size; i++)
        {
            for (int j = 0; j < window_size; j++)
            {
                sum_mask += static_cast<int>(mask.at<uchar>(i, j));
            }
        }

        for (int r = window_size / 2; r < height - window_size / 2; ++r)
        {
            for (int c = window_size / 2; c < width - window_size / 2; ++c)
            {

                // box filter
                int sum = 0;
                for (int i = -window_size / 2; i <= window_size / 2; ++i)
                {
                    for (int j = -window_size / 2; j <= window_size / 2; ++j)
                    {
                        int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
                        int weight = static_cast<int>(mask.at<uchar>(i + window_size / 2, j + window_size / 2));
                        sum += intensity * weight;
                    }
                }
                output.at<uchar>(r, c) = sum / sum_mask;
            }
        }
    }
    void CreateGaussianMask(int window_size, cv::Mat &mask, double sigma)
    {

        const double sigmaSq = sigma * sigma;
        mask = cv::Mat(window_size, window_size, CV_8UC1);

        const int hw = window_size / 2;
        for (int i = 0; i < window_size; ++i)
        {
            for (int j = 0; j < window_size; ++j)
            {
                double r2 = (i - hw) * (i - hw) + (j - hw) * (j - hw);
                mask.at<uchar>(i, j) = 255 * std::exp(-r2 / (2 * sigmaSq));
            }
        }
    }

    void OurGaussianFiler(const cv::Mat &input, cv::Mat &output, int window_size = 5, double sigmaFactor = 1.0)
    {
        // e^-r^2(x, y)/2(sigma^2)
        const auto width = input.cols;
        const auto height = input.rows;

        const int hw = window_size / 2;
        const double sigma = hw / 2.5 * sigmaFactor;

        cv::Mat mask;
        CreateGaussianMask(window_size, mask, sigma);
        // normalize mask
        GeneralFiler(input, output, mask);
    }

    void BilterelarFiler(const cv::Mat &input, cv::Mat &output, int window_size = 5, double sigmaSpatialFactor = 1.0, double sigmaRange = 20.0)
    {
        // e^-r^2(x, y)/2(sigma^2)
        const auto width = input.cols;
        const auto height = input.rows;

        const int hw = window_size / 2;
        const double sigmaSpatial = hw / 2.5 * sigmaSpatialFactor;

        cv::Mat mask;
        CreateGaussianMask(window_size, mask, sigmaSpatial);

        const float sigmaRangeSq = sigmaRange * sigmaRange;

        float range_mask[256];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                output.at<uchar>(i, j) = 0;
            }
        }

        // compute kernel range
        for (int diff = 0; diff < 256; diff++)
        {
            range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
        }

        // normalize mask

        int sum_mask = 0;
        for (int i = 0; i < window_size; i++)
        {
            for (int j = 0; j < window_size; j++)
            {
                sum_mask += static_cast<int>(mask.at<uchar>(i, j));
            }
        }

        for (int r = window_size / 2; r < height - window_size / 2; ++r)
        {
            for (int c = window_size / 2; c < width - window_size / 2; ++c)
            {
                int intensity_center = static_cast<int>(input.at<uchar>(r, c));
                // box filter
                int sum = 0;
                float sumBilateralMask = 0;
                for (int i = -window_size / 2; i <= window_size / 2; ++i)
                {
                    for (int j = -window_size / 2; j <= window_size / 2; ++j)
                    {
                        int intensity = static_cast<int>(input.at<uchar>(r + i, c + j));
                        int diff = std::abs(intensity_center - intensity);
                        float weight_range = range_mask[diff];
                        int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size / 2, j + window_size / 2));
                        float weight = weight_range * weight_spatial;
                        sum += intensity * weight;
                        sumBilateralMask += weight;
                    }
                }
                output.at<uchar>(r, c) = sum / sumBilateralMask;
            }
        }
    }

    void JointBilterelarFiler(const cv::Mat &D, const cv::Mat &I, cv::Mat &output, int window_size = 5, double sigmaSpatialFactor = 1.0, double sigmaRange = 20.0)
    {
        std::cout << "Applying JB with " << sigmaSpatialFactor << std::endl;
        // e^-r^2(x, y)/2(sigma^2)
        const auto width = I.cols;
        const auto height = I.rows;

        const int hw = window_size / 2;
        const double sigmaSpatial = hw / 2.5 * sigmaSpatialFactor;

        cv::Mat mask;
        CreateGaussianMask(window_size, mask, sigmaSpatial);

        cv::Mat output_copy = output.clone();

        const float sigmaRangeSq = sigmaRange * sigmaRange;

        float range_mask[256];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                output_copy.at<uchar>(i, j) = 0;
            }
        }

        // compute kernel range
        for (int diff = 0; diff < 256; diff++)
        {
            range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
        }

        // normalize mask
        int sum_mask = 0;
        for (int i = 0; i < window_size; i++)
        {
            for (int j = 0; j < window_size; j++)
            {
                sum_mask += static_cast<int>(mask.at<uchar>(i, j));
            }
        }

#pragma omp parallel for
        for (int r = window_size / 2; r < height - window_size / 2; ++r)
        {

            for (int c = window_size / 2; c < width - window_size / 2; ++c)
            {
                int intensity_center = static_cast<int>(I.at<uchar>(r, c));
                // box filter
                int sum = 0;
                float sumBilateralMask = 0;
                for (int i = -window_size / 2; i <= window_size / 2; ++i)
                {
                    for (int j = -window_size / 2; j <= window_size / 2; ++j)
                    {
                        int intensity = static_cast<int>(I.at<uchar>(r + i, c + j));
                        int intensity_depth = static_cast<int>(D.at<uchar>(r + i, c + j));
                        int diff = std::abs(intensity_center - intensity);
                        float weight_range = range_mask[diff];
                        int weight_spatial = static_cast<int>(mask.at<uchar>(i + window_size / 2, j + window_size / 2));
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

    void JointBileteralUpsamplingFilter(const cv::Mat &D, const cv::Mat &I, cv::Mat &output, int window_size = 5, double sigmaSpatialFactor = 1.0, double sigmaRange = 20.0)
    {
        cv::Mat D_upsampled;
        cv::resize(D, D_upsampled, I.size(), 0, 0, cv::INTER_NEAREST);
        JointBilterelarFiler(D_upsampled, I, output, window_size, sigmaSpatialFactor, sigmaRange);
        return;
    }

    void IterativeUpsamplingFilter(const cv::Mat &D, const cv::Mat &I, cv::Mat &output, int window_size = 5, double sigmaSpatialFactor = 1.0, double sigmaRange = 20.0)
    {
        int fu = static_cast<int>(std::floor(std::log2(I.rows / D.rows) + 0.00001));

        cv::Mat D_hat = D.clone();
        cv::Mat I_hat;
        for (size_t i = 1; i <= fu - 1; i++)
        {
            cv::resize(D_hat, D_hat, D_hat.size() * 2, 0, 0, cv::INTER_LINEAR);
            cv::resize(I, I_hat, D_hat.size(), 0, 0, cv::INTER_LINEAR);
            JointBilterelarFiler(D_hat, I_hat, D_hat, window_size, sigmaSpatialFactor, sigmaRange);
        }
        cv::resize(D_hat, D_hat, I.size(), 0, 0, cv::INTER_LINEAR);
        JointBilterelarFiler(D_hat, I, output, window_size, sigmaSpatialFactor, sigmaRange);
    }

    void readDataset(const std::string &datasetName, cv::Mat &I_out, cv::Mat &D_out_gt, cv::Mat &D_out, cv::Mat &D_out_downsampled, int &dmin)
    {
        fs::path data_path("data");
        data_path = data_path / datasetName;

        fs::path I_path = data_path / "intensity.png";
        fs::path D_path = data_path / "disp_DP.png";
        fs::path D_gt_path = data_path / "disparity.png";
        fs::path dmin_path = data_path / "dmin.txt";
        fs::path output_dir("output");
        output_dir /= datasetName;

        I_out = cv::imread(I_path.string(), 0);
        if (I_out.data == nullptr)
            std::cerr << "Failed to load image intensity image for " << datasetName << std::endl;
        cv::resize(I_out, I_out, I_out.size()/2, 0, 0, CV_INTER_LINEAR);
        D_out_gt = cv::imread(D_gt_path.string(), 0);
        if (D_out_gt.data == nullptr)
            std::cerr << "Failed to load groundtruth disparity image for " << datasetName << std::endl;
        cv::resize(D_out_gt, D_out_gt, D_out_gt.size()/2, 0, 0, CV_INTER_LINEAR);
        D_out = cv::imread(D_path.string(), 0);
        if (D_out.data == nullptr)
            std::cerr << "Failed to load image disparity image for " << datasetName << std::endl;
        cv::resize(D_out, D_out, D_out.size()/2, 0, 0, CV_INTER_LINEAR);

        std::ifstream MyReadFile(dmin_path.string());
        std::string dmin_string;
        std::getline(MyReadFile, dmin_string);
        dmin = std::stoi(dmin_string);
        MyReadFile.close();

        cv::resize(D_out, D_out_downsampled, D_out_gt.size() / 4, 0, 0, cv::INTER_LINEAR);
        // fs::create_directory("output");
        fs::create_directories(output_dir);
    }

    void Disparity2PointCloud(
        const std::string &output_file, cv::Mat &disparities,
        const int &dmin, const double &baseline = 160, const double &focal_length = 3740)
    {
	int rows = disparities.rows;
	int cols = disparities.cols;
	std::stringstream out3d;
	out3d << output_file << ".xyz";
	std::ofstream outfile(out3d.str());

	for (int r = 0; r < rows; ++r)
	{
		std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((r) / static_cast<double>(rows + 1)) * 100) << "%\r" << std::flush;
		// #pragma omp parallel for
		for (int c = 0; c < cols; ++c)
		{
			if (disparities.at<uchar>(r, c) == 0)
				continue;

			int d = (int)disparities.at<uchar>(r, c) + dmin;
			int u1 = c - cols / 2;
			int u2 = c + d - cols / 2;
			int v1 = r - rows / 2;

			const double Z = baseline * focal_length / d;
			const double X = -0.5 * (baseline * (u1 + u2)) / d;
			const double Y = baseline * v1 / d;
			outfile << X << " " << Y << " " << Z << std::endl;
		}
	}

	std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
	std::cout << std::endl;
}

}


#endif