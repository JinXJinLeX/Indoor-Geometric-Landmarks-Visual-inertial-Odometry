#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

cv::Mat pre_process(cv::Mat image, int color_flag);
int rect_detect(cv::Mat candidate, int color_flag, std::vector<cv::Point2d> &rectangles, double &similarity, double xmin, double ymin);
int tri_detect(cv::Mat candidate, int color_flag, std::vector<cv::Point> &triangles, double &similarity, double xmin, double ymin);
void makecontours(std::vector<cv::Point> points, std::vector<cv::Point> &contours);
double polygonArea(const std::vector<cv::Point> &points);
double cross(const cv::Point2f &p1, const cv::Point2f &p2);
std::vector<cv::Point2f> sortPoints(std::vector<cv::Point> &points, cv::Point center);
int getMinYIndex(const std::vector<cv::Point2f> &points);
int getMaxYIndex(const std::vector<cv::Point2f> &points);
int getMinXIndex(const std::vector<cv::Point2f> &points);
double sign_class_to_id(const std::string input);
std::string sign_id_to_class(const double input);