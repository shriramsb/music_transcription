// helper functions to read/write vector<vector<Point> > to xml file

#include <iostream>
#include "opencv2/core.hpp"

typedef std::vector<std::vector<cv::Point> > vvP;

std::string NumberToString(int i);
void writeVoV(cv::FileStorage &f, std::string name, vvP &contours);
void readVoV(cv::FileStorage &f, std::string name, vvP &contours);

