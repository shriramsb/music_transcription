// helper functions to draw contours and get boundingRectangle of contours 

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

typedef std::vector<std::vector<cv::Point> > vvP;

void draw_contours_show(cv::Mat &img, vvP &contours, bool separate = false);
void get_bddRect(vvP &contours, vvP &contours_poly, std::vector<cv::Rect> &boundRect);