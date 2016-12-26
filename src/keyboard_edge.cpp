/* 	1. Give the image of keyboard and destination xml file to store the corners of keyboard as input
	2. Press i to select the i-th corner of keyboard starting from 0
	3. Make sure that 0-1 and 3-4 edges are perpendicular to keyboard keys
		and 0-1 edge is near the white key end(lower end of keyboard)
*/ 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

struct KeyboardCorners {
	vector<Point2f> corners;
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
	k.corners = vector<Point2f>(4, Point2f(0, 0));
	img = imread(argv[1], 1);
	namedWindow("KeyboardEdges", WINDOW_NORMAL);
	imshow("KeyboardEdges", img);
	setMouseCallback("KeyboardEdges", MouseCallBack, &k);
	char ch = 'i';

	while (ch != 'f') {
		if (ch == 'r') {
			k.no = 0;
		}
		ch = waitKey(0);
		if (ch == '0' || ch == '1' || ch == '2' || ch == '3')
			k.no = (int)(ch - 48);
	}
	FileStorage f(argv[2], FileStorage::WRITE);
	f << "keyboard_corners" << k.corners;
	f.release();
	
}