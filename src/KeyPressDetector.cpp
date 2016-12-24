#include "KeyPressDetector.h"

using std::vector;
using std::cout;
using namespace cv;

KeyPressDetector::KeyPressDetector(Mat &bg, vvP &white_keys, vvP &black_keys, vector<int> &gap_pos) :
 bg(bg.clone()), white_keys(white_keys), black_keys(black_keys), gap_pos(gap_pos) {
	s = bg.size();
	get_bddRect(white_keys, white_keys_poly, white_keys_boundRect);
	get_bddRect(black_keys, black_keys_poly, black_keys_boundRect);
	segmentKeys(white_keys, white_keys_Mat);
	segmentKeys(black_keys, black_keys_Mat);
}

void KeyPressDetector::segmentKeys(vvP &keys, vector<Mat> &keys_Mat) {
	for (int k = 0, l = keys.size(); k < l; k++) {
		keys_Mat.push_back(Mat::zeros(s, CV_8UC1));
		for (int i = 0, n = s.width; i < n; i++)
			for (int j = 0, m = s.height; j < m; j++)
				if (pointPolygonTest(keys[k], Point2f(i, j), false) >= 0)
					keys_Mat[k].at<uchar>(j, i) = 255;
	}
}

void KeyPressDetector::grayDiff(Mat &img1, Mat &img2, Mat &diff) {
	Mat img1_gray, img2_gray;
	cvtColor(img1, img1_gray, CV_BGR2GRAY);		// can be optimized as it is always the same
	cvtColor(img2, img2_gray, CV_BGR2GRAY);
	subtract(img1_gray, img2_gray, diff);
}

void KeyPressDetector::extractHand(Mat &img, Mat &hand_mask, vector<Rect> &hand_boundRect) {
	// getting hand_mask
	static Scalar hsv_l = Scalar(0, 18, 60);
	static Scalar hsv_h = Scalar(20, 255, 255);
	Mat img_hsv = img.clone();
	cvtColor(img_hsv, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, hsv_l, hsv_h, hand_mask);

	// getting boundRect of hand
	vvP hand_contours;
	vector<Vec4i> hand_hierarchy;
	findContours(hand_mask.clone(), hand_contours, hand_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
	vvP hand_contours_approx;
	get_bddRect(hand_contours, hand_contours_approx, hand_boundRect);
	for (int i = 0, j = 0, n = hand_boundRect.size(), key_width = s.width/white_keys.size(); j < n; j++) {
		if (hand_boundRect[i].width < key_width/2) {
			hand_contours.erase(hand_contours.begin() + i);
			hand_boundRect.erase(hand_boundRect.begin() + i);
		}
		else
			i++;
	}

}

void KeyPressDetector::removeHand(Mat &img, Mat &hand_mask, Mat &img_w_hand, bool remove_bottom) {
	static Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	subtract(img, hand_mask, img_w_hand);
	dilate(img_w_hand, img_w_hand, element);
	erode(img_w_hand, img_w_hand, element);
	// erode(img_w_hand, img_w_hand, element);
	if (remove_bottom)
		for (int i = 0, n = s.width; i < n; i++) {
			for (int j = 0, m = s.height/3; j < m; j++) {
				img_w_hand.at<char>(j,i) = 0;
			}
		}
}

void KeyPressDetector::thresholdDiff(Mat &diff, bool is_neg) {
	if (is_neg) {
		// to threshold at the places where there is no black key between two white keys
		for (int i = 0, n = gap_pos.size() - 1; i < n; i++) {
			for (int j = black_keys_boundRect[gap_pos[i]].x, m = black_keys_boundRect[gap_pos[i] + 1].x; j < m; j++)
				for (int k = 2*s.height/6, l = s.height; k < l; k++)
					if (diff.at<uchar>(k, j) > 60)
						diff.at<uchar>(k, j) = 150;
		}
	}
	threshold(diff, diff, 100, 255, CV_THRESH_BINARY);
}

void KeyPressDetector::getKeysNearBox(vector<Rect> &boxes, vector<int> &keys, vector<Rect> &keys_boundRect, bool is_black) {
	for (int i = 0, n = keys_boundRect.size(); i < n; i++) {
		for (int j = 0, m = boxes.size(); j < m; j++) {
			if (is_black) {
				bool necessary_cond = keys_boundRect[i].y <= boxes[j].y + boxes[j].height;
				if (!necessary_cond)
					continue;
			}
			bool case1 = keys_boundRect[i].x >= boxes[j].x && keys_boundRect[i].x <= boxes[j].x + boxes[j].width;
			bool case2 = keys_boundRect[i].x + keys_boundRect[i].width >= boxes[j].x && keys_boundRect[i].x + keys_boundRect[i].width <= boxes[j].x + boxes[j].width;
			bool case3 = boxes[j].x >= keys_boundRect[i].x && boxes[j].x + boxes[j].width <= keys_boundRect[i].x + keys_boundRect[i].width;
			if (case1 || case2 || case3) {
				keys.push_back(i);
				break;
			}
		}
	}
}


void KeyPressDetector::getKeysWithBlob(vector<Mat> &keys_Mat, Mat &diff, vector<int> &keys, vector<Mat> &keys_and_diff) {
	for (int p = 0, q = keys.size(), i = 0; p < q; p++) {
		Mat temp;
		bitwise_and(keys_Mat[keys[i]], diff, temp);
		if (countNonZero(temp) == 0) {
			keys.erase(keys.begin() + i);
		}
		else {
			keys_and_diff.push_back(temp);
			i++;
		}
	}
}

void print(vector<int> &v) {
	for (int i = 0, n = v.size(); i < n; i++) {
		cout << v[i] << ' ';
	}
	cout << '\n';
}

void KeyPressDetector::getKeysWithLargeBlob(vector<Mat> &key_and_diff, vector<Rect> &keys_boundRect, vector<int> &pressed_keys, bool is_black) {
	vector<int> final_pressed_keys;
	for (int p = 0, q = pressed_keys.size(); p < q; p++) {
		int curr_key = pressed_keys[p];
		int key_width = keys_boundRect[curr_key].width;
		vector<vector<Point> > curr_key_blob_contours;
		vector<Vec4i> hie;
		findContours(key_and_diff[p], curr_key_blob_contours, hie, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
		vector<vector<Point> > curr_key_blob_contours_poly;
		vector<Rect> curr_key_blob_contours_boundRect;
		get_bddRect(curr_key_blob_contours, curr_key_blob_contours_poly, curr_key_blob_contours_boundRect);
		for (int i = 0; i < curr_key_blob_contours.size(); ) {
			bool above_threshold;
			if (is_black)
				above_threshold = curr_key_blob_contours_boundRect[i].height > keys_boundRect[curr_key].height/3;
			else
				above_threshold = curr_key_blob_contours_boundRect[i].height > s.height/6 && curr_key_blob_contours_boundRect[i].width > key_width/10;
			if (above_threshold) {
				break;
			}
			else {
				curr_key_blob_contours.erase(curr_key_blob_contours.begin() + i);
				curr_key_blob_contours_boundRect.erase(curr_key_blob_contours_boundRect.begin() + i);
			}
		}
		if (curr_key_blob_contours.size() > 0) {
			final_pressed_keys.push_back(curr_key);
		}
	}
	pressed_keys = final_pressed_keys;
	
}


void KeyPressDetector::detectPressed(Mat &fg, vector<int> &pressed_black_keys, vector<int> &pressed_white_keys, bool show) {
	Mat neg_diff, pos_diff;
	grayDiff(bg, fg, neg_diff);
	grayDiff(fg, bg, pos_diff);
	thresholdDiff(neg_diff, true);
	thresholdDiff(pos_diff, false);

	Mat hand_mask;
	vector<Rect> hand_boundRect;
	extractHand(fg, hand_mask, hand_boundRect);

	Mat neg_diff_w_hand, pos_diff_w_hand;

	removeHand(neg_diff, hand_mask, neg_diff_w_hand, true);
	removeHand(pos_diff, hand_mask, pos_diff_w_hand, false);
	
	getKeysNearBox(hand_boundRect, pressed_white_keys, white_keys_boundRect, false);
	getKeysNearBox(hand_boundRect, pressed_black_keys, black_keys_boundRect, true);


	vector<Mat> white_keys_and_diff;
	vector<Mat> black_keys_and_diff;

	getKeysWithBlob(white_keys_Mat, neg_diff_w_hand, pressed_white_keys, white_keys_and_diff);
	getKeysWithBlob(black_keys_Mat, pos_diff_w_hand, pressed_black_keys, black_keys_and_diff);

	getKeysWithLargeBlob(white_keys_and_diff, white_keys_boundRect, pressed_white_keys, false);
	getKeysWithLargeBlob(black_keys_and_diff, black_keys_boundRect, pressed_black_keys, true);

	if (show) {
		Mat pressed_keys_show = fg.clone();
		for (int i = 0; i < pressed_white_keys.size(); i++) {
			int curr_key = pressed_white_keys[i];
			Point w1 = white_keys[curr_key][0];
			Point w2 = white_keys[curr_key][white_keys[curr_key].size() - 2];
			w2.y = s.height/3;
			rectangle(pressed_keys_show, w1, w2, Scalar(0, 0, 255));	
		}
		for (int i = 0; i < pressed_black_keys.size(); i++) {
			int curr_key = pressed_black_keys[i];
			Point b1 = Point(black_keys_boundRect[curr_key].x, black_keys_boundRect[curr_key].y);
			Point b2 = Point(black_keys_boundRect[curr_key].x + black_keys_boundRect[curr_key].width, black_keys_boundRect[curr_key].y + black_keys_boundRect[curr_key].height);
			rectangle(pressed_keys_show, b1, b2, Scalar(0, 0, 255));	
		}
		imshow("pressed_keys", pressed_keys_show);
	}

}