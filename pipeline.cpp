#include "pipeline.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "roi_detector.h"
#include "symbol_classifier.h"

namespace
{
cv::Mat input(const std::string& path)
{
    cv::Mat img = cv::imread(path);
    if (img.empty())
    {
        std::cout << "Failed to load image\n";
        std::exit(1);
    }
    std::cout << "Size: " << img.cols << " x " << img.rows << std::endl;
    return img;
}

void output(const cv::Mat& img)
{
    cv::imshow("img", img);
    cv::waitKey(0);
}

void processImage(int imageId, const CliConfig& config)
{
    std::string path = config.dataDir + "/" + std::to_string(imageId) + ".jpg";
    std::cout << std::to_string(imageId) << std::endl;
    cv::Mat img = input(path);

    cv::Mat binary = toBinary(img, config.binaryThreshold);
    cv::resize(binary, binary, cv::Size(config.resizeWidth, config.resizeHeight), cv::INTER_LANCZOS4);

    cv::Mat clean = keepLargestComponent(binary);
    cv::Mat skeleton = skeletonize(clean);
    cv::Mat features = detectNonLinear(skeleton);
    cv::Mat grown = growRegions(features, config.closeKernel, config.dilateKernel);

    std::vector<cv::Rect> boxes = extractRegions(
        grown,
        config.minArea,
        config.margin,
        config.mergeIou
    );
    std::vector<Detection> detections = classifyRegions(clean, boxes);

    cv::resize(img, img, cv::Size(config.resizeWidth, config.resizeHeight), cv::INTER_AREA);
    drawCandidates(img, detections);
    //cv::resize(img, img, cv::Size(config.resizeWidth, config.resizeHeight), cv::INTER_AREA);
    output(img);
}
}

void runPipeline(const CliConfig& config)
{
    for (int imageId = config.startImage; imageId <= config.endImage; ++imageId)
        processImage(imageId, config);
}
