#include "roi_detector.h"

namespace
{
double iou(const cv::Rect& a, const cv::Rect& b)
{
    const cv::Rect inter = a & b;
    if (inter.area() <= 0)
        return 0.0;

    const double unionArea = static_cast<double>(a.area() + b.area() - inter.area());
    if (unionArea <= 0.0)
        return 0.0;

    return static_cast<double>(inter.area()) / unionArea;
}

std::vector<cv::Rect> mergeOverlappingBoxes(
    const std::vector<cv::Rect>& inputBoxes,
    double mergeIouThreshold)
{
    std::vector<cv::Rect> boxes = inputBoxes;
    bool mergedAny = true;

    while (mergedAny && boxes.size() > 1)
    {
        mergedAny = false;
        std::vector<bool> removed(boxes.size(), false);
        std::vector<cv::Rect> next;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (removed[i])
                continue;

            cv::Rect current = boxes[i];

            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (removed[j])
                    continue;

                const double overlap = iou(current, boxes[j]);
                if (overlap >= mergeIouThreshold)
                {
                    current = current | boxes[j];
                    removed[j] = true;
                    mergedAny = true;
                }
            }

            next.push_back(current);
        }

        boxes.swap(next);
    }

    return boxes;
}
}

cv::Mat toBinary(const cv::Mat& img, int binaryThreshold)
{
    cv::Mat gray, binary;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, binaryThreshold, 255, cv::THRESH_BINARY_INV);

    return binary;
}




cv::Mat keepLargestComponent(const cv::Mat& binary)
{
    cv::Mat labels, stats, centroids;

    int numLabels = cv::connectedComponentsWithStats(
        binary,
        labels,
        stats,
        centroids
    );

    int largestLabel = 1;
    int largestArea = 0;

    for (int i = 1; i < numLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area > largestArea)
        {
            largestArea = area;
            largestLabel = i;
        }
    }

    cv::Mat result = cv::Mat::zeros(binary.size(), CV_8UC1);

    for (int y = 0; y < labels.rows; y++)
    {
        for (int x = 0; x < labels.cols; x++)
        {
            if (labels.at<int>(y,x) == largestLabel)
                result.at<uchar>(y,x) = 255;
        }
    }

    return result;
}

cv::Mat skeletonize(const cv::Mat& binary)
{
    cv::Mat skeleton;

    cv::ximgproc::thinning(
        binary,
        skeleton,
        cv::ximgproc::THINNING_ZHANGSUEN
    );

    return skeleton;
}


int countNeighbors(const cv::Mat& img, int x, int y)
{
    int count = 0;

    for(int dy = -1; dy <= 1; dy++)
    {
        for(int dx = -1; dx <= 1; dx++)
        {
            if(dx == 0 && dy == 0)
                continue;

            if(img.at<uchar>(y + dy, x + dx) > 0)
                count++;
        }
    }

    return count;
}

cv::Mat detectNonLinear(const cv::Mat& skeleton)
{
    cv::Mat result = cv::Mat::zeros(skeleton.size(), CV_8UC1);

    for(int y = 1; y < skeleton.rows - 1; y++)
    {
        for(int x = 1; x < skeleton.cols - 1; x++)
        {
            if(skeleton.at<uchar>(y,x) == 0)
                continue;

            int neighbors = countNeighbors(skeleton, x, y);

            if(neighbors != 2)
                result.at<uchar>(y,x) = 255;
        }
    }

    return result;
}

cv::Mat growRegions(const cv::Mat& img, int closeKernel, int dilateKernel)
{
    cv::Mat closed;
    cv::Mat result;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(closeKernel, closeKernel)
    );

    cv::morphologyEx(img, closed, cv::MORPH_CLOSE, kernel);

    cv::Mat kernel_d = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(dilateKernel, dilateKernel)
    );

    cv::dilate(closed, result, kernel_d);



    return result;
}




std::vector<cv::Rect> extractRegions(
    const cv::Mat& img,
    int minArea,
    int margin,
    double mergeIouThreshold)
{
    cv::Mat labels, stats, centroids;

    int n = cv::connectedComponentsWithStats(
        img,
        labels,
        stats,
        centroids
    );

    std::vector<cv::Rect> boxes;

    for(int i = 1; i < n; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if(area < minArea)
            continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        int x_new = std::max(0, x - margin);
        int y_new = std::max(0, y - margin);

        int x2 = std::min(img.cols, x + w + margin);
        int y2 = std::min(img.rows, y + h + margin);

        int w_new = x2 - x_new;
        int h_new = y2 - y_new;

        boxes.emplace_back(x_new, y_new, w_new, h_new);
    }

    return mergeOverlappingBoxes(boxes, mergeIouThreshold);
}