#include "time.h"
#include "rect&tri_detect.h"
#include <string>

/*********
主要流程：
1. 对图像进行预处理，保留其中的红色边界部分
2. 对处理结果取最外轮廓
3. 对最外轮廓进行拟合得到三角或矩形
*************/
// using namespace std;
using namespace cv;
static int counter = 0;
static int counter1 = 0;
cv::Mat RGB_extractor(std::vector<cv::Mat> channels)
{
    // 简单的rgb提取
    cv::Mat out;
    // cv:Mat Tmp;
    cv::Mat blue = channels.at(0);
    cv::Mat green = channels.at(1);
    cv::Mat red = channels.at(2);
    out = (red * 3.5 - blue * 1.75 - green * 1.75); // 红色
    // Tmp = out;
    return out;
}

/*图像预处理函数：
1. 对图像进行高斯模糊
2. 对红色通道进行提取
3. 形态学去噪声*/
cv::Mat pre_process(cv::Mat image, int color_flag)
{
    cv::Mat dst, dst_;
    std::vector<cv::Mat> channels; // 图片三通道
    // double sigmax, sinmay;

    // cv::GaussianBlur(image, dst, cv::Size(7, 7), 1, 1);
    // cv::imshow("高斯滤波", dst);
    // cv::waitKey(5);

    // printf("!!!!!!!!%d\n",dst.channels());
    if (color_flag == 1)
    {
        cv::split(image, channels); // 将其分为三个通道
                                    // Result_Temp=channels.at(0);
                                    // channels.at(0)=channels.at(2);
                                    // channels.at(2)=Result_Temp;
                                    // merge(channels, Result_Mat);
    }
    if (color_flag == 2)
    {
        cv::split(image, channels);
        cv::Mat Result_Temp = channels.at(0);
        channels.at(0) = channels.at(2);
        channels.at(2) = Result_Temp;
        // merge(channels, Result_Mat);
    }

    dst_ = RGB_extractor(channels);

    cv::threshold(dst_, dst, 100, 255, cv::THRESH_OTSU);
    // cv::imshow("红色通道", dst);
    // cv::waitKey(5);
    // kernel.ones(10, 10, CV_8UC1);
    // cv::morphologyEx(dst, dst_, CV_MOP_OPEN, kernel);
    // cv::Canny(dst_,dst,50,100);

    // cv::namedWindow("红色通道",0);
    // cv::resizeWindow("红色通道",500,500);
    // cv::imshow("红色通道", dst);
    // cv::waitKey(5);
    return dst;
}

bool findIntersection(const std::vector<cv::Point> con, cv::Point2d &intersection)
{
    // 计算斜率
    cv::Point A = con[0];
    cv::Point B = con[1];
    cv::Point C = con[2];
    cv::Point D = con[3];
    double Cy = C.y + 0.0;
    double Cx = C.x + 0.0;
    double Ay = A.y + 0.0;
    double Ax = A.x + 0.0;
    double By = B.y + 0.0;
    double Bx = B.x + 0.0;
    double Dy = D.y + 0.0;
    double Dx = D.x + 0.0;
    double m1 = (Cy - Ay) / (Cx - Ax); // AC的斜率
    double m2 = (Dy - By) / (Dx - Bx); // BD的斜率
    double b1 = Cy - m1 * Cx;
    double b2 = Dy - m2 * Dx;
    // 如果两条线平行，则没有交点
    if (m1 == m2)
    {
        return false;
    }

    // 计算交点的x坐标
    // double x = (C.y - A.y - m1 * C.x + m1 * A.x) / (m1 - m2);
    // 计算交点的y坐标
    // double y = m1 * (x - A.x) + A.y;

    double x = (b2 - b1) / (m1 - m2);
    double y = m1 * x + b1;

    intersection.x = x;
    intersection.y = y;
    return true;
}

bool compareX(const cv::Point &a, const cv::Point &b)
{
    return a.x < b.x;
}

// 函数：根据y坐标对点进行排序
bool compareY(const cv::Point &a, const cv::Point &b)
{
    return a.y > b.y;
}

std::vector<cv::Point2d> orderPointsClockwise(std::vector<cv::Point2d> &pts)
{
    std::vector<cv::Point2d> sortedPoints = pts;
    std::vector<cv::Point2d> temp_Points;
    int index1, index2;
    double sum, max, min;
    sum = 0;
    max = 0;
    min = 2000;

    for (int i = 1; i < sortedPoints.size(); i++)
    {
        sum = sortedPoints[i].x + sortedPoints[i].y;
        if (sum > max) // 找出右下角
        {
            max = sum;
            index1 = i;
        }
        if (sum < min) // 找出左上角
        {
            min = sum;
            index2 = i;
        }
    }
    for (int i = 1; i < sortedPoints.size(); i++)
    {
        if (i == index1 || i == index2)
            continue;
        temp_Points.push_back(sortedPoints[i]);
        // std::cout << "temp_Points" << std::endl;
    }

    if (temp_Points[0].x < temp_Points[1].x && temp_Points[0].y > temp_Points[1].y) // 若是左下角0右上角1就换序变为左下角1右上角0
    {
        cv::Point2d temp;
        temp = temp_Points[0];
        temp_Points[0] = temp_Points[1];
        temp_Points[1] = temp;
    }
    pts.clear();
    pts.push_back(sortedPoints[0]);      // 中
    pts.push_back(temp_Points[0]);       // 右上
    pts.push_back(sortedPoints[index1]); // 右下
    pts.push_back(temp_Points[1]);       // 左下
    pts.push_back(sortedPoints[index2]); // 左上
    // std::cout << "ptssize" << pts[0] << std::endl;
    // std::cout << "ptssize" << pts[1] << std::endl;
    // std::cout << "ptssize" << pts[2] << std::endl;
    // std::cout << "ptssize" << pts[3] << std::endl;
    // std::cout << "ptssize" << pts[4] << std::endl;
    return pts;
}

int rect_detect(cv::Mat candidate, int color_flag, std::vector<cv::Point2d> &rectangles, double &similarity, double xmin, double ymin)
{
    rectangles.clear();
    int isfind = 0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> contours_;
    // cv::Mat gray_cvColor = cv::Mat::zeros(candidate.size(), CV_8UC1); // 灰色
    cv::Mat pro, dst; // 灰色
    dst = candidate;
    pro = pre_process(candidate, color_flag);
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::morphologyEx(pro, gray_cvColor, CV_MOP_OPEN, kernel);
    cv::findContours(pro, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // cv::noArray(),  cv::RETR_LIST,  cv::CHAIN_APPROX_SIMPLE
    // cv::drawContours(dst, contours, -1, cv::Scalar(255, 0, 0));
    for (auto con : contours)
    {
        auto con_ = con;
        double arc_lenth = 0.1 * cv::arcLength(con, true);
        cv::approxPolyDP(con, con_, arc_lenth, true); // 拟合多边形
        double area = contourArea(con_);
        double total_area = candidate.rows * candidate.cols;

        if (con_.size() == 4 && arc_lenth >= 1 && (area / total_area > 0.2)) // 找出四边形
        {
            contours_.push_back(con_);
            cv::Point2d intersection;
            if (findIntersection(con_, intersection))
            {
                // cv::circle(dst, intersection, 1, cv::Scalar(255, 0, 255), 1);
                // cv::line(dst, con_[0], con_[2], cv::Scalar::all(255), 2);
                // cv::line(dst, con_[1], con_[3], cv::Scalar::all(255), 2);
                rectangles.push_back(cv::Point2d(intersection.x + xmin, intersection.y + ymin));
                for (auto p : con_)
                {
                    rectangles.push_back(cv::Point2d(p.x + xmin, p.y + ymin));
                }
                orderPointsClockwise(rectangles);
                cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
                // cv::Mat contour_img = pro.clone();
                // cv::cvtColor(contour_img, contour_img, cv::COLOR_GRAY2RGB);
                // cv::drawContours(contour_img, contours_, -1, cv::Scalar(0, 156, 221), 2);
                // cv::line(dst, cv::Point2d(rectangles[0].x, rectangles[0].y), cv::Point2d(rectangles[1].x, rectangles[1].y), cv::Scalar(0, 156, 221), 2);
                // cv::line(dst, cv::Point2d(rectangles[1].x, rectangles[1].y), cv::Point2d(rectangles[2].x, rectangles[2].y), cv::Scalar(0, 156, 221), 2);
                // cv::line(dst, cv::Point2d(rectangles[2].x, rectangles[2].y), cv::Point2d(rectangles[3].x, rectangles[3].y), cv::Scalar(0, 156, 221), 2);
                // cv::line(dst, cv::Point2d(rectangles[3].x, rectangles[3].y), cv::Point2d(rectangles[0].x, rectangles[0].y), cv::Scalar(0, 156, 221), 2);
                cv::drawContours(dst, contours_, -1, cv::Scalar(0, 156, 221), 2);
                // cv::imshow("drawcontours", dst);
                // cv::waitKey(5);
                char filename[200];
                char filename_[200];
                // sprintf(filename, "/home/scott/gvins_yolo_output/result1/%d.jpg", counter);
                // sprintf(filename_, "/home/scott/gvins_yolo_output/result1_/%d.jpg", counter);
                // imwrite(filename, dst);
                // imwrite(filename_, pro);
                counter++;
            }
            similarity = 0.9;
            isfind = 1;
        }
    }

    contours_.clear();
    contours.clear();
    return isfind;
    // cv::line(dst, points[0], points[1], cv::Scalar::all(255), 2);
    // cv::line(dst, points[1], points[2], cv::Scalar::all(255), 2);
    // cv::line(dst, points[2], points[0], cv::Scalar::all(255), 2);
    // std::vector<std::vector<cv::Point>> contours;
    // std::vector<std::vector<cv::Point>> contours_;
    // std::vector<cv::Point> points;
    // std::vector<cv::Point> points_;
    // cv::Mat dst = candidate;
    // cv::Mat gray_cvColor = cv::Mat::zeros(candidate.size(), CV_8UC1); // 灰色
    // cv::Mat pro = cv::Mat::zeros(candidate.size(), CV_8UC1);          // 灰色
    // // cv::Mat adad;//灰色
    // cv::Mat adad = cv::Mat::ones(candidate.size(), CV_8UC1); // 彩色
    // adad = cv::Scalar::all(0);
    // cv::Mat adada = cv::Mat::ones(candidate.size(), CV_8UC1); // 彩色
    // adada = cv::Scalar::all(0);

    // pro = pre_process(candidate, color_flag);
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // cv::morphologyEx(pro, gray_cvColor, CV_MOP_OPEN, kernel);

    // cv::findContours(gray_cvColor, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // cv::noArray(),  cv::RETR_LIST,  cv::CHAIN_APPROX_SIMPLE

    // for (auto con : contours)
    // {
    //     std::vector<cv::Point> con_;
    //     double arc_lenth = cv::arcLength(con, true);
    //     cv::approxPolyDP(con, con_, 0.02 * arc_lenth, true);
    //     if (con_.size() == 4 && arc_lenth >= 50)
    //     {
    //         contours_.push_back(con_);
    //         points = con_;
    //     }
    //     makecontours(points, points_);
    //     similarity = cv::matchShapes(con, points_, cv::CONTOURS_MATCH_I1, 0.0);
    //     cv::line(dst, points[0], points[1], cv::Scalar::all(255), 2);
    //     cv::line(dst, points[1], points[2], cv::Scalar::all(255), 2);
    //     cv::line(dst, points[2], points[3], cv::Scalar::all(255), 2);
    //     cv::line(dst, points[3], points[0], cv::Scalar::all(255), 2);
    //     // if (similarity >= 0.5)
    //     //     printf("smilarity of shape: %f \n", similarity);
    // }

    // // cv::drawContours(adad, contours, -1, cv::Scalar::all(255));
    // // cv::drawContours(adada, contours_, -1, cv::Scalar::all(255));
    // // cv::imshow("gray image", gray_cvColor);
    // // cv::imshow("Contours", adad);
    // // cv::imshow("Contours_", adada);
    // // cv::waitKey(5);
    // // cv::waitKey(5);

    // // cv::Mat hierarchy;
    // // findContours(gray_cvColor, contours,hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    // // for (int i = 0; i < contours.size(); i++) {
    // //    drawContours(adad, contours, i, cv::Scalar(0, 0, 255), 1, 8, hierarchy);
    // // }
    // // printf("************%d", contours.size());
    // // cv::imshow("gray image", gray_cvColor);

    // // double cnt_len = cv::arcLength(contours[0], true);
    // // cv::approxPolyDP(contours[0], rectangles,0.02*cnt_len, true);

    // return 0;
}

void makecontours(std::vector<cv::Point> points, std::vector<cv::Point> &contours)
{
    int num = points.size();
    // std::vector<cv::Point> edges;
    for (int i = 0; i < num; ++i)
    {
        cv::Point p1 = points[i];
        cv::Point p2 = points[(i + 1) % num];
        double len = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
        int num_intermediate_points = (int)len;
        for (int j = 0; j < num_intermediate_points; ++j)
        {
            double ratio = (double)j / num_intermediate_points;
            int x = (int)(p1.x + ratio * (p2.x - p1.x));
            int y = (int)(p1.y + ratio * (p2.y - p1.y));
            contours.push_back(cv::Point(x, y));
        }
    }
    return;
}

int tri_detect(cv::Mat candidate, int color_flag, std::vector<cv::Point> &triangles, double &similarity, double xmin, double ymin) // 输入图像，输出三角形顶点、相似度
{
    // double similarity;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> contours_;
    // std::vector<std::vector<cv::Point>> points_contours_;
    std::vector<cv::Point> points;
    std::vector<cv::Point> points_;
    // cv::Mat gray_cvColor = cv::Mat::zeros(candidate.size(), CV_8UC1); // 灰色
    cv::Mat pro;
    cv::Mat dst;
    dst = candidate;
    // cv::Mat adad;//灰色
    // cv::Mat adad = cv::Mat::ones(candidate.size(), CV_8UC1);
    // adad = cv::Scalar::all(0);
    // cv::Mat adada = cv::Mat::ones(candidate.size(), CV_8UC1);
    // adada = cv::Scalar::all(0);
    // cv::Mat adadad = cv::Mat::ones(candidate.size(), CV_8UC1);
    // adadad = cv::Scalar::all(0);

    pro = pre_process(candidate, color_flag);
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::morphologyEx(pro, gray_cvColor, CV_MOP_OPEN, kernel);
    cv::findContours(pro, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // cv::noArray(),  cv::RETR_LIST,  cv::CHAIN_APPROX_SIMPLE

    for (auto con : contours)
    {
        auto con_ = con;
        cv::minEnclosingTriangle(con_, points);                                 // 根据边缘拟合三角形
        makecontours(points, points_);                                          // 画多边形
        similarity = cv::matchShapes(con_, points_, cv::CONTOURS_MATCH_I1, 0.0); // 输入的应该是一个轮廓！
    }
    // 计算多边形的绝对面积
    double area = polygonArea(points_);
    if (similarity < 0.1 && area > 100)
    {
        // printf("*****smilarity of tri shape: %f \n", similarity);
        // printf("*****area of tri shape: %f \n", area);
        // points_contours_.push_back(points_);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
        cv::line(dst, points[0], points[1], cv::Scalar(0, 255, 102), 2);
        cv::line(dst, points[1], points[2], cv::Scalar(0, 255, 102), 2);
        cv::line(dst, points[2], points[0], cv::Scalar(0, 255, 102), 2);
        // 创建一个彩色图像用于绘制轮廓

        char filename[200];
        char filename_[200];
        // sprintf(filename, "/home/scott/gvins_yolo_output/result2/%d.jpg", counter1);
        // sprintf(filename_, "/home/scott/gvins_yolo_output/result2_/%d.jpg", counter1);
        // imwrite(filename, dst);
        // imwrite(filename_, pro);
        counter1++;
        // cv::imshow("tir_con", dst);
        // cv::waitKey(5);
        cv::Point2f center;
        center.x = (points[0].x + points[1].x + points[2].x) / 3;
        center.y = (points[0].y + points[1].y + points[2].y) / 3;
        std::vector<cv::Point2f> sortedPoints = sortPoints(points, center);
        triangles.push_back(cv::Point2f(center.x + xmin, center.y + ymin));
        triangles.push_back(cv::Point2f(sortedPoints[0].x + xmin, sortedPoints[0].y + ymin));
        triangles.push_back(cv::Point2f(sortedPoints[1].x + xmin, sortedPoints[1].y + ymin));
        triangles.push_back(cv::Point2f(sortedPoints[2].x + xmin, sortedPoints[2].y + ymin));
    }
    points.clear();
    points_.clear();
    contours_.clear();
    contours.clear();
    // points_contours_.clear();
    return 0;
}

double polygonArea(const std::vector<cv::Point> &points)
{
    int n = points.size();
    double area = 0.0;
    for (int i = 0; i < n; ++i)
    {
        int j = (i + 1) % n;
        area += cross(points[i], points[j]);
    }
    return fabs(area) / 2.0;
}

double cross(const cv::Point2f &p1, const cv::Point2f &p2)
{
    return p1.x * p2.y - p1.y * p2.x;
}

// 对三个点进行顺时针排序
// 获取y坐标最小的索引
int getMinYIndex(const std::vector<cv::Point> &points)
{
    int minYIndex = 0;
    for (int i = 0; i < points.size(); i++)
    {
        if (points[i].y < points[minYIndex].y)
        {
            minYIndex = i;
        }
    }
    return minYIndex;
}

// 获取y坐标最大的索引
int getMaxYIndex(const std::vector<cv::Point> &points)
{
    int maxYIndex = 0;
    for (int i = 0; i < points.size(); i++)
    {
        if (points[i].y > points[maxYIndex].y)
        {
            maxYIndex = i;
        }
    }
    return maxYIndex;
}

// 获取x坐标最小的索引
int getMinXIndex(const std::vector<cv::Point> &points)
{
    int minXIndex = 0;
    for (int i = 0; i < points.size(); i++)
    {
        if (points[i].x < points[minXIndex].x)
        {
            minXIndex = i;
        }
    }
    return minXIndex;
}

std::vector<cv::Point2f> sortPoints(std::vector<cv::Point> &points, cv::Point center)
{
    // 复制输入的点
    std::vector<cv::Point2f> sortedPoints;
    for (auto p : points)
    {
        sortedPoints.push_back(cv::Point2f(p.x, p.y));
    }
    int counter = 0;
    int index1 = 0;
    int index2 = 0;
    for (auto p : sortedPoints)
    {
        if (p.y < center.y)
        {
            counter++;
        }
    }
    std::cout << "counter" << counter << std::endl;
    if (counter == 1) // 倒三角
    {
        index1 = getMinXIndex(points);
        sortedPoints[0] = points[index1];
        index2 = getMinYIndex(points);
        sortedPoints[1] = points[index2];
        sortedPoints[2] = points[3 - index2 - index1];
    }
    if (counter == 2) // 正三角
    {
        index1 = getMinXIndex(points);
        sortedPoints[0] = points[index1];
        index2 = getMaxYIndex(points);
        sortedPoints[1] = points[index2];
        sortedPoints[2] = points[3 - index2 - index1];
    }
    return sortedPoints;
}

std::string sign_id_to_class(const double input)
{
    // 确定类别前缀
    std::string prefix;
    // 计算实际的数字部分（1-9）
    int number = static_cast<int>(input) % 10;

    // 根据输入数字的范围确定前缀
    if (input >= 11.0 && input <= 19.0)
    {
        prefix = "cir";
    }
    else if (input >= 21.0 && input <= 29.0)
    {
        prefix = "tri";
    }
    else if (input >= 31.0 && input <= 39.0)
    {
        prefix = "rect";
    }
    else
    {
        printf("Input number is out of expected range");
    }

    // 构造并返回结果字符串
    return prefix + "_" + std::to_string(number * 10);
}

double sign_class_to_id(const std::string input)
{
    size_t underscore_pos = input.find('_');
    if (underscore_pos == std::string::npos)
    {
        printf("Input does not contain an underscore");
    }
    std::string prefix = input.substr(0, underscore_pos);

    // 提取数字部分并转换为整数
    std::string number_str = input.substr(underscore_pos + 1);
    int number = (std::stoi(number_str)) / 10;

    // 根据前缀和数字生成对应的输出值
    if (prefix == "cir")
    {
        return 10 + number; // 11 to 19
    }
    else if (prefix == "tri")
    {
        return 20 + number; // 21 to 29
    }
    else if (prefix == "rect")
    {
        return 30 + number; // 31 to 39
    }
    else
    {
        printf("Unknown prefix");
    }
}