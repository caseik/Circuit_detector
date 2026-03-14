#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/ximgproc.hpp>

cv::Mat input(const std::string& path)
{
    cv::Mat img = cv::imread(path);

    if (img.empty())
    {
        std::cout << "Failed to load image\n";
        exit(1);
    }

    return img;
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

cv::Mat toBinary(const cv::Mat& img)
{
    cv::Mat gray, binary;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 120, 255, cv::THRESH_BINARY_INV);

    return binary;
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

std::vector<cv::Point> detectPaper(const cv::Mat& img)
{
    cv::Mat gray, blur, edges;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);
    cv::Canny(blur, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    std::vector<cv::Point> bestContour;

    for (const auto& contour : contours)
    {
        double area = cv::contourArea(contour);

        if (area < 1000)
            continue;

        std::vector<cv::Point> approx;

        cv::approxPolyDP(
            contour,
            approx,
            0.02 * cv::arcLength(contour, true),
            true
        );

        if (approx.size() == 4 && area > maxArea)
        {
            maxArea = area;
            bestContour = approx;
        }
    }

    return bestContour;
}

void output(cv::Mat& img,
            const std::vector<cv::Point>& contour,
            const cv::Mat& binary,
            const cv::Mat& clean)
{
    if (!contour.empty())
    {
        for (int i = 0; i < 4; i++)
        {
            cv::line(
                img,
                contour[i],
                contour[(i + 1) % 4],
                cv::Scalar(0,255,0),
                3
            );
        }
    }

    //cv::imshow("original", img);
    //cv::imshow("binary", binary);
    cv::imshow("clean", clean);

    cv::waitKey(0);
}
cv::Mat growRegions(const cv::Mat& img)
{
    cv::Mat dilated;

    cv::dilate(
        img,
        dilated,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15))
    );

    return dilated;
    
}

std::vector<cv::Rect> extractRegions(const cv::Mat& img)
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

        if(area < 300)
            continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        boxes.push_back(cv::Rect(x,y,w,h));
    }

    return boxes;
}

void drawCandidates(cv::Mat& img, const std::vector<cv::Rect>& boxes)
{
    for(const auto& box : boxes)
    {
        cv::rectangle(
            img,
            box,
            cv::Scalar(0,0,255),
            2
        );
    }
}
int main()
{
    cv::Mat img = input("data/309_png.rf.afb52851067e1c0aa9cc92104fb60ea9.jpg");

    std::vector<cv::Point> paper = detectPaper(img);

    cv::Mat binary = toBinary(img);

    cv::Mat clean = keepLargestComponent(binary);

    cv::Mat skeleton = skeletonize(clean);

    cv::Mat features = detectNonLinear(skeleton);

    cv::Mat grown = growRegions(features);

    std::vector<cv::Rect> boxes = extractRegions(grown);

    drawCandidates(img, boxes);

    output(img, paper, binary, img);
    return 0;
}