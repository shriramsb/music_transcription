#include "my_contours_utility.h"

using std::vector;
using namespace cv;

void draw_contours_show(Mat &img, vvP &contours, bool separate) {
	RNG rng(12345);
	namedWindow("contours", WINDOW_NORMAL);
	Mat img_contour = img.clone();
	for (int i = 0, n = contours.size(); i < n; i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(img_contour, contours, (int)i, color, 2, 8);
		if (separate) {
			imshow("contours", img_contour);
			waitKey(0);
		}
	}
	if (!separate) {
		imshow("contours", img_contour);
		waitKey(0);
	}
	destroyWindow("contours");
}

void get_bddRect(vvP &contours, vvP &contours_poly, vector<Rect> &boundRect) {
	contours_poly = vvP(contours.size());
	boundRect = vector<Rect>(contours.size());
	for (int i = 0, n = contours.size(); i < n; i++) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
}