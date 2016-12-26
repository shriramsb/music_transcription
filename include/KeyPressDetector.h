#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "my_contours_utility.h"

class KeyPressDetector {
	cv::Mat bg;
	cv::Size s;
	vvP white_keys;
	vvP black_keys;
	std::vector<int> gap_pos;
	vvP white_keys_poly;
	vvP black_keys_poly;
	std::vector<cv::Rect> black_keys_boundRect;
	std::vector<cv::Rect> white_keys_boundRect;
	std::vector<cv::Mat> white_keys_Mat;
	std::vector<cv::Mat> black_keys_Mat;
	void segmentKeys(vvP &keys, std::vector<cv::Mat> &keys_Mat);
	void grayDiff(cv::Mat &img1, cv::Mat &img2, cv::Mat &diff);	// grayscale difference
	void thresholdDiff(cv::Mat &diff, bool is_neg);
	void extractHand(cv::Mat &img, cv::Mat &hand_mask, std::vector<cv::Rect> &hand_boundRect);
	void removeHand(cv::Mat &img, cv::Mat &hand_mask, cv::Mat &img_w_hand, bool remove_bottom);
	void getKeysNearBox(std::vector<cv::Rect> &boxes, std::vector<int> &keys, std::vector<cv::Rect> &keys_boundRect, bool is_black);
	void getKeysWithBlob(std::vector<cv::Mat> &keys_Mat, cv::Mat &diff, std::vector<int> &keys, std::vector<cv::Mat> &keys_and_diff);
	void getKeysWithLargeBlob(std::vector<cv::Mat> &key_and_diff, std::vector<cv::Rect> &keys_boundRect, std::vector<int> &pressed_keys, bool is_black);
public:
	KeyPressDetector(cv::Mat &bg_trans, vvP &white_keys, vvP &black_keys, std::vector<int> &gap_pos);
	void detectPressed(cv::Mat &fg, std::vector<int> &pressed_black_keys, std::vector<int> &pressed_white_keys, bool show);

};
