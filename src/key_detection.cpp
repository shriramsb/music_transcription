#include <iostream>
#include <fstream>
#include <stack>
#include <algorithm>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void modify_detected(vector<vector<Point> > &contours, vector<Rect> &boundRect, vector<Moments> &mu, vector<Point2f> &mc, int n_white_keys) {
	
	// getting median in y coordinate of edges detected and average deviation from median
	vector<int> u_y, l_y, dist;
	for (int i = 0, n = contours.size(); i < n; i++) {
		u_y.push_back(boundRect[i].y);
		l_y.push_back(boundRect[i].y + boundRect[i].height);
	}
	cout << "dist_push_back" << endl;
	for (int i = 0, n = contours.size(); i < n - 1; i++) {
		// width/2 need not be added according to the angle
		dist.push_back(boundRect[i + 1].x + boundRect[i + 1].width/2 - boundRect[i].x - boundRect[i].width/2);
	}
	sort(u_y.begin(), u_y.end());
	sort(l_y.begin(), l_y.end());
	sort(dist.begin(), dist.end());
	int u_y_median = u_y.size() % 2 == 0 ? (u_y[u_y.size()/2] + u_y[u_y.size()/2 + 1])/2 : u_y[(u_y.size() + 1)/2];
	int l_y_median = l_y.size() % 2 == 0 ? (l_y[l_y.size()/2] + l_y[l_y.size()/2 + 1])/2 : l_y[(l_y.size() + 1)/2];
	int dist_median = dist.size() % 2 == 0 ? (dist[dist.size()/2] + dist[dist.size()/2 + 1])/2 : dist[(dist.size() + 1)/2];
	vector<int> u_y_error, l_y_error, dist_error;
	for (int i = 0, n = u_y.size(); i < n; i++) {
		u_y_error.push_back(abs(u_y[i] - u_y_median));
		l_y_error.push_back(abs(l_y[i] - l_y_median));
	}
	for (int i = 0, n = dist.size(); i < n; i++) {
		dist_error.push_back(abs(dist[i] - dist_median));
	}
	sort(u_y_error.begin(), u_y_error.end());
	sort(l_y_error.begin(), l_y_error.end());
	sort(dist_error.begin(), dist_error.end());
	int u_y_error_median = u_y_error.size() % 2 == 0 ? (u_y_error[u_y_error.size()/2] + u_y_error[u_y_error.size()/2 + 1])/2 : u_y_error[(u_y_error.size() + 1)/2];
	int l_y_error_median = l_y_error.size() % 2 == 0 ? (l_y_error[l_y_error.size()/2] + l_y_error[l_y_error.size()/2 + 1])/2 : l_y_error[(l_y_error.size() + 1)/2];
	int dist_error_median = dist_error.size() % 2 == 0 ? (dist_error[dist_error.size()/2] + dist_error[dist_error.size()/2 + 1])/2 : dist_error[(dist_error.size() + 1)/2];
	cout << u_y_median << ' ' << l_y_median << ' ' << dist_median << endl;
	cout << u_y_error_median << ' ' << l_y_error_median << ' ' << dist_error_median << endl;

	// correcting detected edges vertically
	for (int i = 0, n = contours.size(); i < n; i++) {
		if (abs(boundRect[i].y - u_y_median) > 10*u_y_error_median || abs(boundRect[i].y + boundRect[i].height - l_y_median) > 10*l_y_error_median) {		
			int x = boundRect[i].x + boundRect[i].width/2;
			contours[i].clear();
			contours[i].push_back(Point(x, u_y_median));
			contours[i].push_back(Point(x, l_y_median));
			boundRect[i] = boundingRect(Mat(contours[i]));
		}
	}

	// adding undetected edges of white keys
	int i = 0;
	cout << "n_white_keys" << n_white_keys << endl;
	while (i < n_white_keys) {

		if (abs(boundRect[i + 1].x + boundRect[i + 1].width/2 - boundRect[i].x - boundRect[i].width/2) > 3.0*dist_median/2) {
			cout << i << ' ' << dist[i] << endl;
			waitKey(0);
			vector<Point> temp;
			temp.push_back(Point(boundRect[i].x + dist_median, u_y_median));
			temp.push_back(Point(boundRect[i].x + dist_median, l_y_median));
			contours.insert(contours.begin() + i + 1, temp);
			boundRect.insert(boundRect.begin() + i + 1, boundingRect(Mat(temp)));
		}
		i++;
	}
}


// sorts contours from left to right using information from boundRect - uses selection sort
void my_sort(vector<vector<Point> > &contours, vector<Rect> &boundRect) {
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

void get_black_keys(Mat &bg, vector<vector<Point> > &black_keys, vector<Rect> &black_keys_boundRect) {
	RNG rng(12345);	
	Mat bg_gray;				// grayscale bg image
	Mat bg_bin;					// for thresholded binary image 
	Mat bg_show = bg.clone();	// for displaying the image
	cvtColor(bg, bg_gray, CV_BGR2GRAY);

	// thresholding to get only the black keys (inverted to make black keys white)
	threshold(bg_gray, bg_bin, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	
	// getting contours of black keys
	vector<vector<Point> > contours;	// vector implemented with list, more efficient
	vector<Vec4i> hierarchy;
	findContours(bg_bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0, n = contours.size(); i < n; i++) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	// cout << contours.size() << endl;
	Size s = bg.size();
	for (int i = 0, n = contours.size(), j = 0; i < n; i++) {
		if (boundRect[j].height < s.height/3 || boundRect[j].width < s.width/100) {
			boundRect.erase(boundRect.begin() + j);
			contours.erase(contours.begin() + j);
		}
		else {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(bg_show, contours, (int)j, color, 2, 8);
			j++;
		}
	}
	imshow("black_keys", bg_show);
	waitKey(0);

	// sorting to get black keys from left to right
	my_sort(contours, boundRect);
	// for (int i = 0; i < boundRect.size(); i++) {
	// 	cout << boundRect[i].x << ' ' << boundRect[i].y << endl;
	// }
	black_keys = contours;
	black_keys_boundRect = boundRect;

}

void get_white_keys(Mat &bg, vector<vector<Point> > &black_keys, vector<Rect> &black_keys_boundRect, vector<vector<Point> > &white_keys) {
	RNG rng(12345);
	
	
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
	for (int i = 0, i_max = bg_grad_x_clipped.size().width; i < i_max; i++)
		for (int j = black_y_avg, j_max = bg_grad_x_clipped.size().height; j < j_max; j++)
			bg_grad_x_clipped.at<uchar>(j, i) = 0;
	imshow("bg_grad_x_clipped", bg_grad_x_clipped);

	// thresholding to get edges of white keys
	Mat bg_grad_x_clipped_threshold;
	int blockSize = bg_grad_x_clipped.size().width;
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
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(bg_grad_x_clipped_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0, n = contours.size(); i < n; i++) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	// removing false positive white keys detected
	// cout << contours.size() << endl;
	Mat bg_lower_contour = bg.clone();
	Size s = bg.size();
	for (int i = 0, n = contours.size(), j = 0; i < n; i++) {
		if (boundRect[j].height < s.height/6 || boundRect[j].width > s.width/10) {
			boundRect.erase(boundRect.begin() + j);
			contours.erase(contours.begin() + j);
		}
		else {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(bg_lower_contour, contours, (int)j, color, 2, 8);
			j++;
		}
	}
	my_sort(contours, boundRect);
	

	// for (int i = 0; i < boundRect.size(); i++) {
	// 	cout << boundRect[i].x << ' ' << boundRect[i].y << endl;
	// }

	// adding white key contours at the edge of the picture if not present
	int avg_white_y = 0;
	for (int i = 0, n = boundRect.size(); i < n; i++)
		avg_white_y += (boundRect[i].y + boundRect[i].height);
	avg_white_y /= boundRect.size();
	if (boundRect[0].x > s.width/100) {
		vector<Point> temp;
		temp.push_back(Point(0, 0));
		temp.push_back(Point(0, avg_white_y));
		contours.insert(contours.begin(), temp);
		boundRect.insert(boundRect.begin(), boundingRect(Mat(contours[0])));
	}
	if (boundRect[boundRect.size() - 1].x < 99*s.width/100) {
		vector<Point> temp;
		temp.push_back(Point(s.width - 1, 0));
		temp.push_back(Point(s.width - 1, avg_white_y));
		contours.push_back(temp);
		boundRect.push_back(boundingRect(Mat(contours[contours.size() - 1])));
	}
	Mat bg_lower_contour_after = bg.clone();
	for (int i = 0, n = contours.size(); i < n; i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(bg_lower_contour_after, contours, (int)i, color, 2, 8);
	}
	imshow("bg_lower_contour_after", bg_lower_contour_after);
	waitKey(0);
	cout << "contours.size()" << contours.size() << endl;
	

	// getting positions where there is no black key between two white keys
	int avg_black_dist = (black_keys_boundRect[black_keys_boundRect.size() - 1].x - black_keys_boundRect[0].x) / (black_keys_boundRect.size() - 1);
	vector<int> gap_pos;
	for (int i = 0, n = black_keys.size() - 1; i < n; i++) {
		if (black_keys_boundRect[i + 1].x - black_keys_boundRect[i].x > avg_black_dist)
			gap_pos.push_back(i);
	}
	gap_pos.push_back(black_keys.size() - 1);
	for (int i = 0, n = gap_pos.size(); i < n; i++)
		cout << gap_pos[i] << ' ';
	cout << endl;

	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	for (int i = 0, n = contours.size(); i < n; i++) {
		mu[i] = moments(contours[i], false);
		mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);
	}


	// checking if white key edges are properly detected and correcting if necessary
	modify_detected(contours, boundRect, mu, mc, black_keys.size() + gap_pos.size());

	Mat after_modify_detect = bg.clone();
	for (int i = 0, n = contours.size(); i < n; i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(after_modify_detect, contours, (int)i, color, 2, 8);
		imshow("after_modify_detect", after_modify_detect);
		cout << boundRect[i].x << endl;
		waitKey(0);
	}
	// polygon approximation of contours (can try convexHull)
	vector<vector<Point> > black_keys_poly(black_keys.size());
	for (int i = 0, n = black_keys.size(); i < n; i++) {
		approxPolyDP(Mat(black_keys[i]), black_keys_poly[i], 3, true);
	}

	// contours of white keys
	white_keys = vector<vector<Point> >(contours.size() - 1);
	bool gap_left = false;
	bool gap_right = true;
	int k = 0;
	cout << "ok" << endl;
	cout << contours.size() << endl;
	cout << white_keys.size() << endl;
	
	for (int i = 0, n = white_keys.size(); i < n; i++) {
		if(gap_left) {
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, boundRect[i].y));
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, boundRect[i].y + boundRect[i].height));
			int j = 0;
			while (j < black_keys_poly[k - 1].size() && black_keys_poly[k - 1][j].x > boundRect[i].x)
				j++;
			cout << (contourArea(black_keys_poly[k - 1], true) > 0) << endl;
			if (contourArea(black_keys_poly[k - 1],true) > 0)
				for (int p = 0, m = black_keys_poly[k - 1].size(); p < m; p++) {
					if (black_keys_poly[k - 1][(p + j)%m].x > boundRect[i].x)
						white_keys[i].push_back(black_keys_poly[k - 1][(p+j)%m]);
				}
			else {
				stack<Point> t;
				for (int p = 0, m = black_keys_poly[k - 1].size(); p < m; p++) {
					if (black_keys_poly[k - 1][(p+j)%m].x > boundRect[i].x)
						t.push(black_keys_poly[k - 1][(p+j)%m]);
				}
				while (!t.empty()) {
					white_keys[i].push_back(t.top());
					t.pop();
				}
			}

			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, s.height));
			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, boundRect[i + 1].y));
			gap_left = false;
			gap_right = true;
		}
		else if (gap_right) {
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, boundRect[i].y));
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, s.height));
			int j = 0;
			cout << "ok" << endl;
			while (j < black_keys_poly[k].size() && black_keys_poly[k][j].x < boundRect[i + 1].x)
				j++;
			cout << (contourArea(black_keys_poly[k], true) > 0) << endl;
			if (contourArea(black_keys_poly[k],true) > 0)
				for (int p = 0, m = black_keys_poly[k].size(); p < m; p++) {
					if (black_keys_poly[k][(p+j)%m].x < boundRect[i + 1].x)
						white_keys[i].push_back(black_keys_poly[k][(p+j)%m]);
				}
			else {
				stack<Point> t;
				for (int p = 0, m = black_keys_poly[k].size(); p < m; p++) {
					if (black_keys_poly[k][(p+j)%m].x < boundRect[i + 1].x)
						t.push(black_keys_poly[k][(p+j)%m]);
				}
				while (!t.empty()) {
					white_keys[i].push_back(t.top());
					t.pop();
				}
			}
			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, boundRect[i + 1].y + boundRect[i + 1].height));
			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, boundRect[i + 1].y));
			gap_right = false;
			k++;
		}
		else {
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, boundRect[i].y));
			white_keys[i].push_back(Point(boundRect[i].x + boundRect[i].width/2, boundRect[i].y + boundRect[i].height));
			int j = 0;
			while (j < black_keys_poly[k - 1].size() && black_keys_poly[k - 1][j].x > boundRect[i].x)
				j++;
			cout << (contourArea(black_keys_poly[k - 1], true) > 0) << endl;
			if (contourArea(black_keys_poly[k - 1],true) > 0)
				for (int p = 0, m = black_keys_poly[k - 1].size(); p < m; p++) {
					if (black_keys_poly[k - 1][(p + j)%m].x > boundRect[i].x)
						white_keys[i].push_back(black_keys_poly[k - 1][(p+j)%m]);
					else
						break;
				}
			else {
				stack<Point> t;
				for (int p = 0, m = black_keys_poly[k - 1].size(); p < m; p++) {
					if (black_keys_poly[k - 1][(p+j)%m].x > boundRect[i].x)
						t.push(black_keys_poly[k - 1][(p+j)%m]);
				}
				while (!t.empty()) {
					white_keys[i].push_back(t.top());
					t.pop();
				}
			}
			j = 0;
			while (j < black_keys_poly[k].size() && black_keys_poly[k][j].x < boundRect[i + 1].x)
				j++;
			cout << (contourArea(black_keys_poly[k], true) > 0) << endl;
			if (contourArea(black_keys_poly[k],true) > 0)
				for (int p = 0, m = black_keys_poly[k].size(); p < m; p++) {
					if (black_keys_poly[k][(p+j)%m].x < boundRect[i + 1].x)
						white_keys[i].push_back(black_keys_poly[k][(p+j)%m]);
				}
			else {
				stack<Point> t;
				for (int p = 0, m = black_keys_poly[k].size(); p < m; p++) {
					if (black_keys_poly[k][(p+j)%m].x < boundRect[i + 1].x)
						t.push(black_keys_poly[k][(p+j)%m]);
				}
				while (!t.empty()) {
					white_keys[i].push_back(t.top());
					t.pop();
				}
			}
			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, boundRect[i + 1].y + boundRect[i + 1].height));
			white_keys[i].push_back(Point(boundRect[i + 1].x + boundRect[i + 1].width/2, boundRect[i + 1].y));
			k++;
			if (!gap_pos.empty() && (k - 1) == gap_pos.front()) {
				gap_left = true;
				gap_pos.erase(gap_pos.begin());
			}

		}
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(bg, white_keys, (int)i, color, 2, 8);
		imshow("white_keys", bg);
		waitKey(0);
	}
	for (int i = 0, n = white_keys.size(); i < n; i++) {
		
	}
	imshow("white_keys", bg);
	waitKey(0);



}

int main(int argc, char** argv) {
	Mat bg;
	bg = imread(argv[1], 1);
	vector<vector<Point> > black_keys;
	vector<Rect> black_keys_boundRect;
	get_black_keys(bg, black_keys, black_keys_boundRect);
	// cout << black_keys.size() << endl;
	vector<vector<Point> > white_keys;
	get_white_keys(bg, black_keys, black_keys_boundRect, white_keys);


}