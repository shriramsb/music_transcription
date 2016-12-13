/*
Give input image as command line argument
Press '0' and click anywhere to select the first corner
and then press '1' and click on the image and so on
and after selecting all four corners, press 'f' to finish and save corners to file "keyboard_edge.xml"
*/

#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

struct KeyboardCorners {
	vector<Point2d> corners;
	int no;
};

Mat img;
Mat img_with_corners;

void MouseCallBack(int event, int x, int y, int flags, void* userdata) {
	KeyboardCorners* p = (KeyboardCorners*)userdata;
	if (event == EVENT_LBUTTONDOWN) {
		p->corners[p->no].x = x;
		p->corners[p->no].y = y;
		img_with_corners = img.clone();
		for (int i = 0; i <= p->no; i++) {
			circle(img_with_corners, p->corners[i], 3, Scalar(0, 0, 255));
		}
		for (int i = 0; i < p->no; i++) {
			line(img_with_corners, p->corners[i], p->corners[i + 1], Scalar(0, 255, 0));
		}
		if (p->no == 3)
			line(img_with_corners, p->corners[3], p->corners[0], Scalar(0, 255, 0));
		imshow("KeyboardEdges", img_with_corners);
	}
}

int main(int argc, char** argv) {

	KeyboardCorners k;
	k.no = 0;
	k.corners = vector<Point2d>(4, Point2d(0, 0));
	img = imread(argv[1], 1);
	namedWindow("KeyboardEdges", WINDOW_NORMAL);
	imshow("KeyboardEdges", img);
	setMouseCallback("KeyboardEdges", MouseCallBack, &k);
	char ch = 'i';

	while (ch != 'f') {
		if (ch == 'h') {
			cout << "r - reset\nf - finish"; 
		}
		else if (ch == 'r') {
			k.no = 0;
		}
		ch = waitKey(0);
		if (ch == '0' || ch == '1' || ch == '2' || ch == '3')
			k.no = (int)(ch - 48);
	}
	FileStorage f("keyboard_edge.xml", FileStorage::WRITE);

	f << "corner0" << k.corners[0];
	f << "corner1" << k.corners[1];
	f << "corner2" << k.corners[2];
	f << "corner3" << k.corners[3];
	f.release();
	
}