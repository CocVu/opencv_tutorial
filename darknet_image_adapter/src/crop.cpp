extern "C"
{
#include "image.h"
}
#include "blob.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
int main(int argc, char* argv[])
{
  string image_file_name = "../image/origin.png";
  cv::Mat img = cv::imread(image_file_name);
  if (img.empty())
    {
      std::cout << "!!! imread() failed to open target image" << std::endl;
      return -1;
    }

  /* Set Region of Interest */

  int offset_x = 40;
  int offset_y = 40;

  cv::Rect roi;
  roi.x = offset_x;
  roi.y = offset_y;
  roi.width = img.size().width - (offset_x*2);
  roi.height = img.size().height - (offset_y*2);

  cv::Mat crop = img(roi);
  cv::imshow("crop", crop);
  cv::waitKey(0);

  cv::imwrite("output/crop.png", crop);

  return 0;
}
