#include <opencv2/opencv.hpp>
#include "input/input.h"
#include "pipeline/pipeline.h"




int main()
{
    cv::Mat img = cv::imread("data/272_png.rf.b6462912dcb455f71c0300271bdfc231.jpg");

    if (img.empty())
    {
        std::cout << "Failed to load image\n";
        return 1;
    }

    cv::imshow("image", img);
    cv::waitKey(0);

    return 0;
}