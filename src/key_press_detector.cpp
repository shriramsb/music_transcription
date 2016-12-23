/*	1. Takes video, background_transfomed image, xml containing transform matrix and xml containing white and black keys
		as input
	2. Displays the pressed keys
*/

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "my_contours_utility.h"
#include "filehandle_vov.h"

using std::cout;
using std::vector;
using namespace cv;

Size s;

void get_hand_mask(Mat &fg, Mat &hand_mask) {
	static Scalar hsv_l = Scalar(0, 18, 60);
	static Scalar hsv_h = Scalar(20, 255, 255);
	Mat fg_hsv = fg.clone();
	cvtColor(fg_hsv, fg_hsv, CV_BGR2HSV);
	inRange(fg_hsv, hsv_l, hsv_h, hand_mask);
}

void get_bw_diff(Mat &bg, Mat &fg, Mat &diff) {
	Mat bg_gray, fg_gray;
	cvtColor(bg, bg_gray, CV_BGR2GRAY);		// can be optimized as it is always the same
	cvtColor(fg, fg_gray, CV_BGR2GRAY);
	subtract(bg_gray, fg_gray, diff);
}

void get_keys_mat(vvP &keys, vector<Mat> &keys_mat) {
	for (int k = 0, l = keys.size(); k < l; k++) {
		keys_mat.push_back(Mat::zeros(s, CV_8UC1));
		for (int i = 0, n = s.width; i < n; i++)
			for (int j = 0, m = s.height; j < m; j++)
				if (pointPolygonTest(keys[k], Point2f(i, j), false) >= 0)
					keys_mat[k].at<uchar>(j, i) = 255;
	}
}

void get_diff_w_hand(Mat &diff, Mat &hand_mask, Mat &diff_w_hand, bool remove_bottom) {
	static Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	subtract(diff, hand_mask, diff_w_hand);
	dilate(diff_w_hand, diff_w_hand, element);
	erode(diff_w_hand, diff_w_hand, element);
	// erode(diff_w_hand, diff_w_hand, element);
	if (remove_bottom)
		for (int i = 0, n = s.width; i < n; i++) {
			for (int j = 0, m = s.height/3; j < m; j++) {
				diff_w_hand.at<char>(j,i) = 0;
			}
		}
}

bool is_key_near(vector<Point> &key, Rect &boundRect) {
	bool case1 = key[0].x >= boundRect.x && key[0].x <= boundRect.x + boundRect.width;
	if (case1)
		return true;
	bool case2 = key[key.size() - 1].x >= boundRect.x && key[key.size() - 1].x <= boundRect.x + boundRect.width;
	if (case2)
		return true;
	bool case3 = boundRect.x >= key[0].x && boundRect.x + boundRect.width <= key[key.size() - 1].x;
	if (case3)
		return true;
	return false;
	
}

void keys_near_hand(vvP &keys, vector<Rect> &hand_boundRect, vector<int> &pressed_keys) {
	for (int i = 0, n = keys.size(); i < n; i++) {
		for (int j = 0, m = hand_boundRect.size(); j < m; j++) {
			if (is_key_near(keys[i], hand_boundRect[j])) {
				pressed_keys.push_back(i);
				break;
			}
		}
	}
}

void print(vector<int> &v) {
	for (int i = 0, n = v.size(); i < n; i++) {
		cout << v[i] << ' ';
	}
	cout << '\n';
}

int main(int argc, char** argv) {
	VideoCapture cap(argv[1]);
	Mat bg;			// background image
	Mat fg;			// frame
	Mat neg_diff;	// bg - fg
	Mat pos_diff;	// fg - bg

	bg = imread(argv[2], 1);
	s = bg.size();
	
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

	vector<vector<Point> > black_keys_poly;
	vector<Rect> black_keys_boundRect;
	get_bddRect(black_keys, black_keys_poly, black_keys_boundRect);


	vector<Mat> white_keys_mat;
	vector<Mat> black_keys_mat;
	get_keys_mat(white_keys, white_keys_mat);
	get_keys_mat(black_keys, black_keys_mat);


	int wait = 30;
	while (true) {
		cap >> fg;
		Mat fg_trans;
		warpPerspective(fg, fg_trans, t, s);

		get_bw_diff(bg, fg_trans, neg_diff);
		get_bw_diff(fg_trans, bg, pos_diff);

		// to threshold at the places where there is no black key between two white keys
		for (int i = 0, n = gap_pos.size() - 1; i < n; i++) {
			for (int j = black_keys_boundRect[gap_pos[i]].x, m = black_keys_boundRect[gap_pos[i] + 1].x; j < m; j++)
				for (int k = 2*s.height/6, l = s.height; k < l; k++)
					if (neg_diff.at<uchar>(k, j) > 60)
						neg_diff.at<uchar>(k, j) = 150;
		}

		// imshow("neg_diff", neg_diff);
		// imshow("pos_diff", pos_diff);

		threshold(neg_diff, neg_diff, 100, 255, CV_THRESH_BINARY);
		threshold(pos_diff, pos_diff, 100, 255, CV_THRESH_BINARY);
		imshow("neg_diff_after_threshold", neg_diff);
		imshow("pos_diff_after_threshold", pos_diff);


		Mat hand_mask;
		get_hand_mask(fg_trans, hand_mask);
		imshow("hand_mask", hand_mask);


		// dilate(hand_mask, hand_mask, element);
		// Mat hand;
		// fg_trans.copyTo(hand, hand_mask);
		// imshow("hand", hand);
		
		
		Mat neg_diff_w_hand;
		get_diff_w_hand(neg_diff, hand_mask, neg_diff_w_hand, true);
		imshow("neg_diff_w_hand", neg_diff_w_hand);
		
		
		Mat pos_diff_w_hand;
		get_diff_w_hand(pos_diff, hand_mask, pos_diff_w_hand, false);
		imshow("pos_diff_w_hand", pos_diff_w_hand);


		vector<int> pressed_keys;

		// bounding box for hand - step 1
		Mat hand_cont_img = fg_trans.clone();
		vvP hand_contours;
		vector<Vec4i> hand_hierarchy;
		Mat temp = hand_mask.clone();
		findContours(temp, hand_contours, hand_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
		vvP hand_contours_approx;
		vector<Rect> hand_boundRect;
		get_bddRect(hand_contours, hand_contours_approx, hand_boundRect);
		for (int i = 0, j = 0, n = hand_boundRect.size(), key_width = s.width/white_keys.size(); j < n; j++) {
			if (hand_boundRect[i].width < key_width/2) {
				hand_contours.erase(hand_contours.begin() + i);
				hand_boundRect.erase(hand_boundRect.begin() + i);
			}
			else {
				rectangle(hand_cont_img, Point(hand_boundRect[i].x, hand_boundRect[i].y), Point(hand_boundRect[i].x + hand_boundRect[i].width, hand_boundRect[i].y + hand_boundRect[i].height), Scalar(0, 0, 255));
				i++;
			}
		}
		imshow("hand_cont_img", hand_cont_img);
		keys_near_hand(white_keys, hand_boundRect, pressed_keys);

		cout << "inside bdd_rect" << ' ';
		print(pressed_keys);

		// key and contours
		vector<Mat> key_and_contour;
		for (int p = 0, q = pressed_keys.size(), i = 0; p < q; p++) {
			Mat temp;
			bitwise_and(white_keys_mat[pressed_keys[i]], neg_diff_w_hand, temp);
			if (countNonZero(temp) == 0) {
				pressed_keys.erase(pressed_keys.begin() + i);
			}
			else {
				key_and_contour.push_back(temp);
				i++;
			}
		}

		cout << "Non zero" << ' ';
		print(pressed_keys);

		vector<int> final_pressed_keys;

		for (int p = 0, q = pressed_keys.size(); p < q; p++) {
			int curr_key = pressed_keys[p];
			int key_width = white_keys[curr_key][white_keys[curr_key].size() - 1].x - white_keys[curr_key][0].x;
			vector<vector<Point> > contours_pressed_key;
			vector<Vec4i> hie;
			findContours(key_and_contour[p], contours_pressed_key, hie, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
			vector<vector<Point> > contours_pressed_key_poly;
			vector<Rect> contours_pressed_key_boundRect;
			get_bddRect(contours_pressed_key, contours_pressed_key_poly, contours_pressed_key_boundRect);
			for (int i = 0; i < contours_pressed_key.size(); ) {
				if (contours_pressed_key_boundRect[i].height > s.height/6 && contours_pressed_key_boundRect[i].width > key_width/10) {
					break;
				}
				else {
					contours_pressed_key.erase(contours_pressed_key.begin() + i);
					contours_pressed_key_boundRect.erase(contours_pressed_key_boundRect.begin() + i);
				}
			}
			if (contours_pressed_key.size() > 0) {
				final_pressed_keys.push_back(curr_key);
			}
		}
		Mat pressed_keys_show = fg_trans.clone();
		for (int i = 0; i < final_pressed_keys.size(); i++) {
			line(pressed_keys_show, white_keys[final_pressed_keys[i]][0], white_keys[final_pressed_keys[i]][1], Scalar(0, 0, 255));
			line(pressed_keys_show, white_keys[final_pressed_keys[i]][white_keys[final_pressed_keys[i]].size() - 2], white_keys[final_pressed_keys[i]][white_keys[final_pressed_keys[i]].size() - 1], Scalar(0, 0, 255));
		}
		imshow("pressed_keys_show", pressed_keys_show);




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