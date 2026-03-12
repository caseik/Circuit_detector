#include "input.h"

class VideoInput : public InputSource
{
private:
    cv::VideoCapture cap;

public:

    VideoInput(int camera_id = 0)
    {
        cap.open(camera_id);
    }

    bool getFrame(cv::Mat& frame) override
    {
        return cap.read(frame);
    }
};