/*	1. Takes transformed image and destination xml file for storing detected keys as input
	2. Stores the contours of black and white keys in the xml file
*/

#include <iostream>
#include <stack>
#include <algorithm>
#include "filehandle_vov.h"
#include "my_contours_utility.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using std::vector;
using std::stack;
using std::string;
using std::ostringstream;
using namespace cv;


void gapPosition(vector<Rect> &black_keys_boundRect, vector<int> &gap_pos) {
	int avg_black_dist = (black_keys_boundRect[black_keys_boundRect.size() - 1].x - black_keys_boundRect[0].x) / (black_keys_boundRect.size() - 1);
	for (int i = 0, n = black_keys_boundRect.size() - 1; i < n; i++) {
		if (black_keys_boundRect[i + 1].x - black_keys_boundRect[i].x > avg_black_dist)
			gap_pos.push_back(i);
	}
	gap_pos.push_back(black_keys_boundRect.size() - 1);
}


int median(vector<int> &list) {
	sort(list.begin(), list.end());
	int l = list.size();
	return (l % 2 == 0 ? (list[l/2] + list[l/2 + 1])/2 : list[(l+1)/2]);
}

int medianDev(vector<int> &list, int list_median) {
	int l = list.size();
	vector<int> list_dev(l);
	for (int i = 0; i < l; i++) {
		list_dev[i] = abs(list[i] - list_median);
	}
	return median(list_dev);
}


// add/change improperly detected contours
void modifyContours(int pos, int x, int uy, int ly, vvP &contours, vector<Rect> &boundRect, vector<Moments> &mu, vector<Point2f> &mc, bool add_new) {
	vector<Point> temp;
	temp.push_back(Point(x, uy));
	temp.push_back(Point(x, ly));
	if (add_new) {
		contours.insert(contours.begin() + pos, temp);
		boundRect.insert(boundRect.begin() + pos, boundingRect(Mat(temp)));
		mu.insert(mu.begin() + pos, moments(temp, false));
		mc.insert(mc.begin() + pos, Point2f(x, (uy + ly)/2));
	}
	else {
		contours[pos].clear();
		contours[pos] = temp;
		boundRect[pos] = boundingRect(Mat(contours[pos]));
		mu[pos] = moments(contours[pos], false);
		mc[pos] = Point2f(x, (uy + ly)/2);
	}
}

// sorts contours from left to right using information from boundRect - uses selection sort
void sortContours(vvP &contours, vector<Rect> &boundRect) {
	for (int i = 0, n = contours.size(); i < n; i++) {
		int minIndex = i;
		for (int j = i + 1; j < n; j++) {
			if (boundRect[j].x < boundRect[minIndex].x)
				minIndex = j;
		}
		vector<Point> temp = contours[i];
		contours[i] = contours[minIndex];
		contours[minIndex] = temp;
		Rect temp1 = boundRect[i];
		boundRect[i] = boundRect[minIndex];
		boundRect[minIndex] = temp1;
	}
}

void refineDetected(vvP &contours, vector<Rect> &boundRect, vector<Moments> &mu, vector<Point2f> &mc, int n_white_keys, Mat &bg) {
	Size s = bg.size();

	// removing false positive white keys detected
	Mat bg_lower_contour = bg.clone();
	for (int i = 0, n = contours.size(), j = 0; i < n; i++) {
		if (boundRect[j].height < s.height/6 || boundRect[j].width > s.width/10) {
			boundRect.erase(boundRect.begin() + j);
			contours.erase(contours.begin() + j);
		}
		else
			j++;
	}
	sortContours(contours, boundRect);

	// getting moments and center of mass of detected contours
	for (int i = 0, n = contours.size(); i < n; i++) {
		mu[i] = moments(contours[i], false);
		mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);
	}

	// removing two very close contour
	for (int i = 0, n = contours.size(), j = 0; i < n; i++) {
		if (mc[j + 1].x - mc[j].x < s.width/100) {
			contours.erase(contours.begin() + j + 1);
			boundRect.erase(boundRect.begin() + j + 1);
			mu.erase(mu.begin() + j + 1);
			mc.erase(mc.begin() + j + 1);
		}
		else
			j++;
	}

	// adding white key contours at the edge of the picture if not present
	int avg_white_lower_y = 0;
	for (int i = 0, n = boundRect.size(); i < n; i++)
		avg_white_lower_y += (boundRect[i].y + boundRect[i].height);
	avg_white_lower_y /= boundRect.size();
	if (boundRect[0].x > s.width/100) 
		modifyContours(0, 0, 0, avg_white_lower_y, contours, boundRect, mu, mc, true);
	if (boundRect[boundRect.size() - 1].x < 99*s.width/100)
		modifyContours(contours.size(), s.width - 1, 0, avg_white_lower_y, contours, boundRect, mu, mc, true);


	// getting median in y coordinate of edges detected and average deviation from median
	vector<int> u_y, l_y, dist;
	for (int i = 0, n = contours.size(); i < n; i++) {
		u_y.push_back(boundRect[i].y);
		l_y.push_back(boundRect[i].y + boundRect[i].height);
	}
	for (int i = 0, n = contours.size(); i < n - 1; i++)
		dist.push_back(mc[i + 1].x - mc[i].x);

	int u_y_median = median(u_y);
	int l_y_median = median(l_y);
	int dist_median = median(dist); 
	int u_y_error_median = medianDev(u_y, u_y_median);
	int l_y_error_median = medianDev(l_y, l_y_median);


	// correcting detected edges vertically
	for (int i = 0, n = contours.size(); i < n; i++)
		if (abs(boundRect[i].y - u_y_median) > 10*u_y_error_median || abs(boundRect[i].y + boundRect[i].height - l_y_median) > 10*l_y_error_median)	
			modifyContours(i, mc[i].x, u_y_median, l_y_median, contours, boundRect, mu, mc, false);


	// adding undetected edges of white keys
	int i = 0;
	while (i < n_white_keys) {
		if (abs(mc[i + 1].x - mc[i].x) > 3.0*dist_median/2)
			modifyContours(i + 1, mc[i].x + dist_median, u_y_median, l_y_median, contours, boundRect, mu, mc, true);
		i++;
	}
}


void getBlackKeys(Mat &bg, vvP &black_keys, vector<Rect> &black_keys_boundRect) {
	
	Size s = bg.size();
	Mat bg_gray;				// grayscale bg image
	Mat bg_bin;					// for thresholded binary image 
	cvtColor(bg, bg_gray, CV_BGR2GRAY);

	// thresholding to get only the black keys (inverted to make black keys white)
	threshold(bg_gray, bg_bin, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	
	// getting contours of black keys
	vvP contours;	// vector implemented with list, more efficient
	vector<Vec4i> hierarchy;
	findContours(bg_bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
	vvP contours_poly;
	vector<Rect> boundRect;
	get_bddRect(contours, contours_poly, boundRect);
	// cout << contours.size() << endl;
	
	// removing bad contours
	for (int i = 0, n = contours.size(), j = 0; i < n; i++) {
		if (boundRect[j].height < s.height/3 || boundRect[j].width < s.width/100) {
			boundRect.erase(boundRect.begin() + j);
			contours.erase(contours.begin() + j);
		}
		else
			j++;
	}

	// sorting to get black keys from left to right
	sortContours(contours, boundRect);
	
	black_keys = contours;
	black_keys_boundRect = boundRect;

}


// getting contour of white keys from contours of already detected black keys
void transferPoints(vector<Point> &dest, vector<Point> &src, float x_threshold, bool left_side) {
	int side = left_side ? -1 : 1;
	int j = 0;
	while (j < src.size() && side*src[j].x > side*x_threshold)
		j++;
	if (contourArea(src,true) > 0)
		for (int p = 0, m = src.size(); p < m; p++) {
			if (side*src[(p + j)%m].x > side*x_threshold)
				dest.push_back(src[(p+j)%m]);
		}
	else {
		stack<Point> t;
		for (int p = 0, m = src.size(); p < m; p++) {
			if (side*src[(p+j)%m].x > side*x_threshold)
				t.push(src[(p+j)%m]);
		}
		while (!t.empty()) {
			dest.push_back(t.top());
			t.pop();
		}
	}
}

void getWhiteKeys(Mat &bg, vvP &black_keys, vector<Rect> &black_keys_boundRect, vvP &white_keys) {

	Size s = bg.size();

	// extracting white key lines using sobel
	Mat bg_gray;
	cvtColor(bg, bg_gray, CV_BGR2GRAY);
	Mat bg_grad_x;
	Sobel(bg_gray, bg_grad_x, -1, 1, 0);

	// removing edges of black keys captured in sobel
	int black_y_avg = 0;
	for (int i = 0, n = black_keys_boundRect.size(); i < n; i++)
		black_y_avg += black_keys_boundRect[i].y;
	black_y_avg /= black_keys_boundRect.size();
	Mat bg_grad_x_clipped = bg_grad_x.clone();
	for (int i = 0; i < s.width; i++)
		for (int j = black_y_avg; j < s.height; j++)
			bg_grad_x_clipped.at<uchar>(j, i) = 0;
	imshow("bg_grad_x_clipped", bg_grad_x_clipped);

	// thresholding to get edges of white keys -- has to be changed
	Mat bg_grad_x_clipped_threshold;
	int blockSize = s.width;
	if (blockSize % 2 == 0)
		blockSize += 1;
	adaptiveThreshold(bg_grad_x_clipped, bg_grad_x_clipped_threshold, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, 0);
	imshow("bg_grad_x_clipped_threshold", bg_grad_x_clipped_threshold);
	waitKey(0);
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
	dilate(bg_grad_x_clipped_threshold, bg_grad_x_clipped_threshold, element);
	erode(bg_grad_x_clipped_threshold, bg_grad_x_clipped_threshold, element);
	erode(bg_grad_x_clipped_threshold, bg_grad_x_clipped_threshold, element);
	imshow("bg_grad_x_clipped_threshold", bg_grad_x_clipped_threshold);
	waitKey(0);


	// getting contours of white keys edges
	vvP contours;
	vector<Vec4i> hierarchy;
	findContours(bg_grad_x_clipped_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
	vvP contours_poly;
	vector<Rect> boundRect;
	get_bddRect(contours, contours_poly, boundRect);

	// getting positions where there is no black key between two white keys
	vector<int> gap_pos;
	gapPosition(black_keys_boundRect, gap_pos);

	// centers of detected contours
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());

	// checking if white key edges are properly detected and correcting if necessary
	refineDetected(contours, boundRect, mu, mc, black_keys.size() + gap_pos.size(), bg);

	// polygon approximation of contours (can try convexHull)
	vvP black_keys_poly(black_keys.size());
	for (int i = 0, n = black_keys.size(); i < n; i++) {
		approxPolyDP(Mat(black_keys[i]), black_keys_poly[i], 3, true);
	}

	// contours of white keys
	white_keys = vvP(contours.size() - 1);
	bool gap_left = false;
	bool gap_right = true;
	int k = 0;
	
	for (int i = 0, n = white_keys.size(); i < n; i++) {
		if(gap_left) {
			white_keys[i].push_back(Point(mc[i].x, boundRect[i].y));
			white_keys[i].push_back(Point(mc[i].x, boundRect[i].y + boundRect[i].height));
			
			transferPoints(white_keys[i], black_keys_poly[k - 1], mc[i].x, false);

			white_keys[i].push_back(Point(mc[i + 1].x, s.height));
			white_keys[i].push_back(Point(mc[i + 1].x, boundRect[i + 1].y));
			gap_left = false;
			gap_right = true;
		}
		else if (gap_right) {
			white_keys[i].push_back(Point(mc[i].x, boundRect[i].y));
			white_keys[i].push_back(Point(mc[i].x, s.height));
			
			transferPoints(white_keys[i], black_keys_poly[k], mc[i + 1].x, true);
			
			white_keys[i].push_back(Point(mc[i + 1].x, boundRect[i + 1].y + boundRect[i + 1].height));
			white_keys[i].push_back(Point(mc[i + 1].x, boundRect[i + 1].y));
			gap_right = false;
			k++;
		}
		else {
			white_keys[i].push_back(Point(mc[i].x, boundRect[i].y));
			white_keys[i].push_back(Point(mc[i].x, boundRect[i].y + boundRect[i].height));
			
			transferPoints(white_keys[i], black_keys_poly[k - 1], mc[i].x, false);

			transferPoints(white_keys[i], black_keys_poly[k], mc[i + 1].x, true);
			
			white_keys[i].push_back(Point(mc[i + 1].x, boundRect[i + 1].y + boundRect[i + 1].height));
			white_keys[i].push_back(Point(mc[i + 1].x, boundRect[i + 1].y));
			k++;
			if (!gap_pos.empty() && (k - 1) == gap_pos.front()) {
				gap_left = true;
				gap_pos.erase(gap_pos.begin());
			}

		}
	}
}



int main(int argc, char** argv) {
	Mat bg;
	bg = imread(argv[1], 1);
	vvP black_keys;
	vector<Rect> black_keys_boundRect;
	vector<int> gap_pos;
	
	getBlackKeys(bg, black_keys, black_keys_boundRect);
	gapPosition(black_keys_boundRect, gap_pos);

	vvP white_keys;
	getWhiteKeys(bg, black_keys, black_keys_boundRect, white_keys);

	draw_contours_show(bg, black_keys, true);
	draw_contours_show(bg, white_keys, true);
	

	FileStorage f(argv[2], FileStorage::WRITE);
	writeVoV(f, "black_keys", black_keys);
	writeVoV(f, "white_keys", white_keys);
	f << "gap_pos" << gap_pos;
	f.release();

}