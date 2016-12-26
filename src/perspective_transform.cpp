/*	1. Takes image of keyboard and output of keyboard_edge.cpp as input
	2. Also give destination of output XML file for transform matrix to be stored
		and destination of output transformed image to be stored
*/

#include "my_utility.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using std::vector;
using namespace cv;

// gives mappings of keyboard corners from old image to new image
void get_keyboard_mappings(vector<Point2f> &kc_transformed) {
	kc_transformed = vector<Point2f>(4);
	kc_transformed[0].x = kc_transformed[0].y = kc_transformed[1].y = kc_transformed[3].x = 0;
	kc_transformed[1].x = kc_transformed[2].x = KEYBOARD_WIDTH;
	kc_transformed[2].y = kc_transformed[3].y = KEYBOARD_HEIGHT;
}


int main(int argc, char** argv) {
	Mat img;
	Mat transformed_img;
	Mat t;						// to store transform matrix from original image to transformed image
	img = imread(argv[1], 1);
	
	vector<Point2f> kc_actual;
	FileStorage f(argv[2], FileStorage::READ);
	f["keyboard_corners"] >> kc_actual;
	f.release();

	vector<Point2f> kc_transformed(4);		// corners of keyboard in transformed image
	get_keyboard_mappings(kc_transformed);

	t = getPerspectiveTransform(kc_actual, kc_transformed);

	FileStorage f1(argv[3], FileStorage::WRITE);
	f1 << "transform" << t;
	f1.release();

	warpPerspective(img, transformed_img, t, Size(KEYBOARD_WIDTH, KEYBOARD_HEIGHT));
	namedWindow("transformed_img", WINDOW_NORMAL);
	namedWindow("img", WINDOW_NORMAL);
	imshow("img", img);
	imshow("transformed", transformed_img);
	imwrite(argv[4], transformed_img);
	waitKey(0);

}