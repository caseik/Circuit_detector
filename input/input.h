#pragma once
#include <opencv2/opencv.hpp>

class InputSource
{
public:
    virtual bool getFrame(cv::Mat& frame) = 0;
    virtual ~InputSource() {}
};