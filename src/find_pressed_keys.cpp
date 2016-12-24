/*	1. Takes video, background_transfomed image, xml containing transform matrix and xml containing white and black keys
		as input
	2. Displays the pressed keys
*/

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "KeyPressDetector.h"
#include "filehandle_vov.h"


using std::cout;
using std::vector;
using namespace cv;

int main(int argc, char** argv) {
	VideoCapture cap(argv[1]);
	Mat bg_trans = imread(argv[2], 1);
	Mat fg, fg_trans;
	Size s = bg_trans.size();

	Mat t;
	FileStorage f1(argv[3], FileStorage::READ);
	f1["transform"] >> t;
	f1.release();

	FileStorage f(argv[4], FileStorage::READ);
	vvP black_keys;
	vvP white_keys;
	vector<int> gap_pos;
	readVoV(f, "black_keys", black_keys);
	readVoV(f, "white_keys", white_keys);
	f["gap_pos"] >> gap_pos;
	f.release();

	KeyPressDetector detector(bg_trans, white_keys, black_keys, gap_pos);

	int wait = 30;

	namedWindow("pressed_keys", WINDOW_NORMAL);

	while (true) {
		cap >> fg;
		warpPerspective(fg, fg_trans, t, s);
		vector<int> pressed_white_keys;
		vector<int> pressed_black_keys;
		detector.detectPressed(fg_trans, pressed_black_keys, pressed_white_keys, true);
		char q = waitKey(wait);
		if (q == 'p') {
			wait = 0;
		}
		else if (q == 'r') {
			wait = 30;
		}
		else if (q == 'f') {
			wait = 1;
		}

	}

}