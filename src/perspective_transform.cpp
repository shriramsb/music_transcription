/*
Takes image as command line argument using output of keyboard_edge.cpp, transforms the input image to get only the keyboard
*/


#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat img;
Mat dest_img;
Mat t;
int main(int argc, char** argv) {
	img = imread(argv[1], 1);
	vector<Point2f> kc_actual(4);
	FileStorage f("keyboard_edge.xml", FileStorage::READ);
	f["corner0"] >> kc_actual[0];
	f["corner1"] >> kc_actual[1];
	f["corner2"] >> kc_actual[2];
	f["corner3"] >> kc_actual[3];
	for (int i = 0; i < 4; i++) {
		cout << kc_actual[i].x << ' ' << kc_actual[i].y << endl;
	}
	vector<Point2f> kc_transformed(4);
	kc_transformed[0].x = kc_transformed[0].y = kc_transformed[1].y = kc_transformed[3].x = 0;
	kc_transformed[1].x = kc_transformed[2].x = 900;
	kc_transformed[2].y = kc_transformed[3].y = 300;
	t = getPerspectiveTransform(kc_actual, kc_transformed);
	FileStorage f1("test1/transform.xml", FileStorage::WRITE);
	f1 << "transform" << t;
	f1.release();
	warpPerspective(img, dest_img, t, Size(900, 300));
	imshow("a", dest_img);
	imwrite("images/cropped.jpg", dest_img);
	waitKey(0);

}