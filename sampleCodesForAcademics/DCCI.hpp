#pragma once
#include <opencv2/opencv.hpp>
#include <Halide.h>

void DCCI32FC1(const cv::Mat& src_, cv::Mat& dst, const float threshold, int ompNumThreads);
void DCCI32FC1Halide(Halide::Buffer<float>& input, Halide::Buffer<float>& output, float threshold);
