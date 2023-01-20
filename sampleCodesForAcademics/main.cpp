#include "DCCI.hpp"

// OpenCV lib
#define CV_LIB_PREFIX "opencv_"
#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib"
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib"
#endif
#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX
#pragma comment(lib, CV_LIBRARY(core))
#pragma comment(lib, CV_LIBRARY(imgproc))
#pragma comment(lib, CV_LIBRARY(imgcodecs))
#pragma comment(lib, CV_LIBRARY(highgui))

// Halide lib
#pragma comment(lib, "Halide.lib")

using namespace std;
using namespace cv;
using namespace Halide;

void convertHalide2Mat(const Halide::Buffer<float>& src, cv::Mat& dest);
void convertMat2Halide(cv::Mat& src, Halide::Buffer<float>& dest);

int main()
{
	Mat srcMatCpp8UC1 = imread("kodim01.png", 0);
	Size srcSize(srcMatCpp8UC1.cols, srcMatCpp8UC1.rows);
	Size destSize(srcMatCpp8UC1.cols * 2, srcMatCpp8UC1.rows * 2);

    // C++実装の動作確認
	Mat srcMatCpp32FC1(srcSize, CV_32FC1);
	Mat destMatCpp32FC1(destSize, CV_32FC1);
	srcMatCpp8UC1.convertTo(srcMatCpp32FC1, CV_32FC1, 1. / 255.);
	DCCI32FC1(srcMatCpp32FC1, destMatCpp32FC1, 1.15f, 4);
	imshow("C++", destMatCpp32FC1);

    // Halide実装の動作確認
	Buffer<float> srcBufferHalide32FC1(srcSize.width, srcSize.height);
	Buffer<float> destBufferHalide32FC1(destSize.width, destSize.height);
    convertMat2Halide(srcMatCpp32FC1, srcBufferHalide32FC1);
	DCCI32FC1Halide(srcBufferHalide32FC1, destBufferHalide32FC1, 1.15f);
	Mat destMatHalide32FC1(destSize, CV_32FC1);
    convertHalide2Mat(destBufferHalide32FC1, destMatHalide32FC1);
    imshow("Halide", destMatHalide32FC1);
    waitKey();
}

void convertHalide2Mat(const Halide::Buffer<float>& src, cv::Mat& dest)
{
    const int ch = dest.channels();
    if (ch == 1)
    {
        for (int j = 0; j < dest.rows; j++)
        {
            for (int i = 0; i < dest.cols; i++)
            {
                dest.at<float>(j, i) = src(i, j);
            }
        }
    }
    else if (ch == 3)
    {
        for (int j = 0; j < dest.rows; j++)
        {
            for (int i = 0; i < dest.cols; i++)
            {
                dest.at<float>(j, 3 * i + 0) = src(i, j, 0);
                dest.at<float>(j, 3 * i + 1) = src(i, j, 1);
                dest.at<float>(j, 3 * i + 2) = src(i, j, 2);
            }
        }
    }
}

void convertMat2Halide(cv::Mat& src, Halide::Buffer<float>& dest)
{
    const int ch = src.channels();
    if (ch == 1)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                dest(i, j) = src.at<float>(j, i);
            }
        }
    }
    else if (ch == 3)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                dest(i, j, 0) = src.at<float>(j, 3 * i);
                dest(i, j, 1) = src.at<float>(j, 3 * i + 1);
                dest(i, j, 2) = src.at<float>(j, 3 * i + 2);
            }
        }
    }
}
