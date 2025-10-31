#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <math.h>

#include "feature_tracker.h"

// 20230423_xjl
#include <mutex>
#include <queue>
#include "yolov5_ros_msgs/BoundingBox.h"
#include "yolov5_ros_msgs/BoundingBoxes.h"
#include "ellipse_detect.h"
#include "rect&tri_detect.h"

#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match, pub_ellipse, pub_tri, pub_rect, pub_sign;
// 20230920_xjl
ros::Publisher pub_camerainfo;

ros::Publisher pub_restart;
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
double initial_time;
int pub_count = 1;
int pub_ellipse_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

/**************************/
// 20221009_xjl
double detect_freq = 0.98;
double pre_image_time;
double cur_image_time;
bool pub_info = false;
// 20230424_xjl
std::vector<yolov5_ros_msgs::BoundingBox> signal_boxes;
// std::deque<std::pair<string, cv::Mat>> e_sign;
// std::deque<std::pair<string, cv::Mat>> t_sign;
// std::deque<std::pair<string, cv::Mat>> r_sign;
int e_xmin, e_xmax, e_ymin, e_ymax;
int t_xmin, t_xmax, t_ymin, t_ymax;
int r_xmin, r_xmax, r_ymin, r_ymax;
string last_class_of_e, last_class_of_t, last_class_of_r;
// std::deque<std::pair<cv::Mat,double>> img_buf_;
// std::pair<cv::Mat,double> imgwt;
// std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> signal_boxes;

// 20230211xjl
double last_imu_t;
std::mutex m_buf;
queue<IMU> ins_window;
IMU cur_imu;
IMU pre_imu;
double imu_datadt = 0.006; // 200HZ
bool first_imu = 1;
// int datarate = 100; // 100HZ
// cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 892.982544, 0.000000, 611.368633, 0.000000, 900.910278, 353.958833, 0, 0, 1);
cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 952.011, 0.000000, 661.733, 0.000000, 959.650, 375.598, 0, 0, 1);
cv::Mat distortion;
// cv::Mat distortion = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0);
// cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);

// void rect_detect_callback(const sensor_msgs::ImageConstPtr &img_msg)
// {
//     TicToc t_r;
//     int a = 0;
//     int color_flag = 0;
//     cv::Mat img;
//     cv_bridge::CvImageConstPtr ptr;
//     // cout << img_msg->encoding << endl;
//     if (img_msg->encoding == "8UC1")
//     {
//         sensor_msgs::Image img_;
//         img_.header = img_msg->header;
//         img_.height = img_msg->height;
//         img_.width = img_msg->width;
//         img_.is_bigendian = img_msg->is_bigendian;
//         img_.step = img_msg->step;
//         img_.data = img_msg->data;
//         img_.encoding = "mono8";
//         ptr = cv_bridge::toCvCopy(img_, sensor_msgs::image_encodings::MONO8);
//         color_flag = 3;
//     }
//     else if (img_msg->encoding == "bgr8")
//     {
//         img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
//         memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
//         // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
//         // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
//         color_flag = 1;
//     }
//     else if (img_msg->encoding == "rgb8")
//     {
//         img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
//         memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
//         // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
//         // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
//         color_flag = 2;
//     }
//     else
//     {
//         ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); // 灰度图
//         img = ptr->image;
//         color_flag = 3;
//     }
// }
// void tri_detect_callback(const sensor_msgs::ImageConstPtr &img_msg)
// {
//         TicToc t_r;
//     int a = 0;
//     int color_flag = 0;
//     cv::Mat img;
//     cv_bridge::CvImageConstPtr ptr;
//     // cout << img_msg->encoding << endl;
//     if (img_msg->encoding == "8UC1")
//     {
//         sensor_msgs::Image img_;
//         img_.header = img_msg->header;
//         img_.height = img_msg->height;
//         img_.width = img_msg->width;
//         img_.is_bigendian = img_msg->is_bigendian;
//         img_.step = img_msg->step;
//         img_.data = img_msg->data;
//         img_.encoding = "mono8";
//         ptr = cv_bridge::toCvCopy(img_, sensor_msgs::image_encodings::MONO8);
//         color_flag = 3;
//     }
//     else if (img_msg->encoding == "bgr8")
//     {
//         img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
//         memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
//         // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
//         // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
//         color_flag = 1;
//     }
//     else if (img_msg->encoding == "rgb8")
//     {
//         img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
//         memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
//         // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
//         // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
//         color_flag = 2;
//     }
//     else
//     {
//         ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); // 灰度图
//         img = ptr->image;
//         color_flag = 3;
//     }
//     std::vector<cv::Point2f> points;
//    tri_detect(img,points);
// }

bool rectanglesIntersect(const std::vector<Point2f> &rect1, const std::vector<Point2f> &rect2)
{
    // 创建两个矩形对象
    Rect_<float> r1 = boundingRect(rect1);
    Rect_<float> r2 = boundingRect(rect2);

    // 判断两个矩形是否相交
    return (r1 & r2).area() > 0;
}

/**************************/
void ellipse_detect_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if (first_image_flag)
    {
        pre_image_time = img_msg->header.stamp.toSec();
        cur_image_time = img_msg->header.stamp.toSec();
        return;
    }
    cur_image_time = img_msg->header.stamp.toSec();
    if (cur_image_time - pre_image_time < detect_freq)
    {
        return;
    }
    // if(signal_boxes.empty())
    // {
    //     return;
    // }
    TicToc t_r;
    int a = 0;
    int color_flag = 0;
    cv::Mat img;
    cv_bridge::CvImageConstPtr ptr;
    // cout << img_msg->encoding << endl;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img_;
        img_.header = img_msg->header;
        img_.height = img_msg->height;
        img_.width = img_msg->width;
        img_.is_bigendian = img_msg->is_bigendian;
        img_.step = img_msg->step;
        img_.data = img_msg->data;
        img_.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img_, sensor_msgs::image_encodings::MONO8);
        color_flag = 3;
    }
    else if (img_msg->encoding == "bgr8")
    {
        img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
        memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        color_flag = 1;
    }
    else if (img_msg->encoding == "rgb8")
    {
        img = Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
        memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        color_flag = 2;
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); // 灰度图
        img = ptr->image;
        color_flag = 3;
    }
    double similarity = 0.0;
    vector<vector<double>> output;
    a = ellipse_detect(img, color_flag, output, similarity, 0.0, 0.0);

    /**********************************************************/
    // 计算圆心在相机系中的位置
    //  20221031_xjl
    // our
    // cv::Mat intrinsic = (Mat_<double>(3, 3) << 1765.682901, 0.000000, 782.352086, 0.000000, 1758.799034, 565.999397, 0, 0, 1);
    // cv::Mat distortion = (Mat_<double>(1, 4) << -0.060942, 0.058542, 0.001478, 0.002002);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);

    std::vector<cv::Point2f> dis_pts;
    Point2f tmp_pts;
    std::vector<cv::Point2f> un_dis_pts;
    int ellipse_num = 0;
    ellipse_num = output.size();

    if (ellipse_num != 0)
    {
        for (int i = 0; i < ellipse_num; i++)
        {
            tmp_pts.x = output[i][0];
            tmp_pts.y = output[i][1];
            // output.pop_back();
            dis_pts.push_back(tmp_pts);
        }
        // 只对圆心做去畸变，长轴短轴不做
        un_dis_pts=dis_pts;
        // cv::undistortPoints(dis_pts, un_dis_pts, intrinsic, distortion, cv::Mat(), intrinsic);
        // dis_pts.clear();
        ROS_WARN("un_dis_pts : %f ,%f", un_dis_pts.back().x, un_dis_pts.back().y);
        // std::cout << "un_dis_pts" << un_dis_pts << std::endl;
    }
    /**********************************************************/

    /***********************************/
    // 20221027xjl
    if (ellipse_num != 0)
    {
        sensor_msgs::PointCloudPtr ellipse_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 u_of_point; //
        sensor_msgs::ChannelFloat32 v_of_point; // 像素坐标(u,v)
        sensor_msgs::ChannelFloat32 a_of_point; // 长轴
        sensor_msgs::ChannelFloat32 b_of_point; // 短轴
        sensor_msgs::ChannelFloat32 ori_of_point;
        // sensor_msgs::ChannelFloat32 dis_of_point;
        for (int i = 0; i < ellipse_num; i++)
        {
            pub_ellipse_count++; // 检测到椭圆的id（暂时没有用到）

            ellipse_points->header = img_msg->header;
            ellipse_points->header.frame_id = "world";
            geometry_msgs::Point32 p;
            p.x = un_dis_pts[i].x;
            p.y = un_dis_pts[i].y;
            p.z = 1; // 矫正后归一化平面的3d点(x,y,1)

            // double dis_of_ellipse;
            // 求距离，输入内参，归一化坐标和长轴
            // dis_of_ellipse = calcu_dis(un_dis_pts[i].x, un_dis_pts[i].y, intrinsic, output[i][2], output[i][3], output[i][4]);
            // point3d real_pos;
            // calcu_p(dis_pts[i].x, dis_pts[i].y, intrinsic, dis_of_ellipse,real_pos);
            // ROS_WARN( "X_of_ellipse : %f",real_pos.x);
            // ROS_WARN( "Y_of_ellipse : %f",real_pos.y );
            // ROS_WARN( "Z_of_ellipse : %f",real_pos.r );

            ellipse_points->points.push_back(p);
            u_of_point.values.push_back(output[i][0]);
            v_of_point.values.push_back(output[i][1]);
            a_of_point.values.push_back(output[i][2]);
            b_of_point.values.push_back(output[i][3]);
            ori_of_point.values.push_back(output[i][4]);
            // dis_of_point.values.push_back(dis_of_ellipse);
            // ROS_WARN( "dis_of_ellipse : %f",dis_of_ellipse );
            // ROS_WARN( "width_of_ellipse : %f",output[i][2] );

            ellipse_points->channels.push_back(u_of_point);
            ellipse_points->channels.push_back(v_of_point);
            ellipse_points->channels.push_back(a_of_point);
            ellipse_points->channels.push_back(b_of_point);
            ellipse_points->channels.push_back(ori_of_point);
            // ellipse_points->channels.push_back(dis_of_point);
            // ROS_WARN("publish %f, at %f", ellipse_points->header.stamp.toSec(), ros::Time::now().toSec());
            pub_ellipse.publish(ellipse_points);
        }
    }

    // if (SHOW_ELLIPSE)
    // {
    //     ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
    //     //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
    //     cv::Mat stereo_img = ptr->image;
    //     cv::Mat show_img = ptr->image;

    // for (int i = 0; i < NUM_OF_CAM; i++)
    // {
    //     cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
    //     cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);

    // for (unsigned int j = 0; j < ellipse_num; j++)
    // {
    // double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
    // cv::ellipse( tmp_img, cv::Point2d(output[j][0],output[j][1]), cv::Size(output[j][2],output[j][3]),output[j][4], 0, 360, CV_RGB(0, 0, 255), 1, cv::LINE_AA, 0);
    // cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    //     //                 //draw speed line
    //     //                 /*
    //     //                 Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
    //     //                 Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
    //     //                 Vector3d tmp_prev_un_pts;
    //     //                 tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
    //     //                 tmp_prev_un_pts.z() = 1;
    //     //                 Vector2d tmp_prev_uv;
    //     //                 trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
    //     //                 cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
    //     //                 */
    // }
    // }
    //     pub_match.publish(ptr->toImageMsg());
    // }
    // }
    /***********************************/

    pre_image_time = cur_image_time;
    ROS_DEBUG("pub ellipse costs: %fms", t_r.toc());
    return;
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // ROS_INFO("processing camera %f",img_msg->header.stamp.toSec());
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        initial_time = first_image_time;
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }

    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    // 将图像的编码格式从8UC1转换成mono8
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    // 存入img缓存 用于对应yolov5的结果
    // imgwt.first = show_img;
    // imgwt.second = img_msg->header.stamp.toSec();
    // img_buf_.emplace_back(imgwt);

    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        // 单目情况
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec(), ins_window); // track在此实现，则先验也在此实现
        // trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec()); // 读取图片并光流跟踪
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img); // 均衡化
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // 更新全局id，将新提取的特征构点赋予全局id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i); // 更新特征点id
        if (!completed)
            break;
    }

    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point; // 特征点id
        sensor_msgs::ChannelFloat32 u_of_point;  // 像素坐标(u,v)
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point; // 像素的速度(vx, vy)
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1; // 矫正后归一化平面的3d点(x,y,1)

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        // ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    // draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    // char name[10];
                    // sprintf(name, "%d", trackerData[i].ids[j]);
                    // cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            // cv::imshow("vis", stereo_img);
            // cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
    // if (!pub_info)
    // {
    sensor_msgs::CameraInfo caminfo;
    caminfo.header = img_msg->header;
    caminfo.header.frame_id = "camera";
    caminfo.height = img_msg->height;
    caminfo.width = img_msg->width;
    caminfo.distortion_model = "plumb_bob";

    // caminfo.D = {-0.055379, 0.051226, 0.000408, -0.002483, 0.000000};
    // caminfo.K = {387.495931, 0.000000, 324.667951, 0.000000, 386.608929, 251.872588, 0.000000, 0.000000, 1.000000}; // 1765.682901, 0.000000, 782.352086, 0, 1758.799034, 565.999397, 0, 0, 1
    // caminfo.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    // caminfo.P = {387.240631, 0.000000, 321.687063, 0.000000, 0.000000, 387.311676, 251.179550, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000};
    caminfo.D = {-0.049547, 0.012867, -0.000750, -0.000176, 0.000000};
    caminfo.K = {619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0.000000, 0.000000, 1.000000}; // 1765.682901, 0.000000, 782.352086, 0, 1758.799034, 565.999397, 0, 0, 1
    caminfo.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    caminfo.P = {619.523712, 0.000000, 656.497684, 0.000000, 0.000000, 615.410395, 403.222400, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000};
    caminfo.binning_x = 0;
    caminfo.binning_y = 0;
    caminfo.roi.x_offset = 0;
    caminfo.roi.y_offset = 0;
    caminfo.roi.height = 0;
    caminfo.roi.width = 0;
    caminfo.roi.do_rectify = false;

    pub_camerainfo.publish(caminfo);
    pub_info = true;
    // }
}

// 20230619_xjl
// 输入图像,类别字符串
// cv::Mat processH(cv::Mat img, string cob)
// {
//     string tmp_str;
//     cv::Mat sign;
//     char filename[200];
//     sprintf(filename, "/home/scott/ellipse/FinalProject_EDbyLMedS/result/g/%d.jpg", cob);
//     sign = cv::imread(filename);
// }

// 20230814_xjl
void judge(int box_xmin, int box_xmax, int box_ymin, int box_ymax, int &xmin, int &xmax, int &ymin, int &ymax)
{
    // xmin = max(0, box_xmin - 50);
    // xmax = min(1440, box_xmax + 50);
    // ymin = max(0, box_ymin - 50);
    // ymax = min(1080, box_ymax + 50);
    xmin = max(0, box_xmin - 20);
    xmax = min(1280, box_xmax + 20);
    ymin = max(0, box_ymin - 20);
    ymax = min(800, box_ymax + 20);
}

// 20230423_xjl
void boxes_callback(const yolov5_ros_msgs::BoundingBoxes &boxes_msg)
{
    // if (boxes_msg.header.stamp.toSec() - initial_time < 5.0) // 前五秒内不检测标志
    //     return;
    // printf("\033[1;36m detect traffic sign !!!! time is: %.9f\n    \033[0m", boxes_msg.image_header.stamp.toSec());
    int a = 0;
    int color_flag = 0;
    int sign_num = 0;
    cv::Mat img;
    // static cv::Mat last_ellipse, last_triangle, last_rectangle;

    sign_num = boxes_msg.bounding_boxes.size();

    // cout << boxes_msg.encoding << endl;
    cv_bridge::CvImageConstPtr ptr;
    if (boxes_msg.encoding == "bgr8")
    {
        img = Mat(static_cast<int>(boxes_msg.height), static_cast<int>(boxes_msg.width), CV_8UC3);
        memcpy(img.data, boxes_msg.data.data(), boxes_msg.height * boxes_msg.width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        color_flag = 1;
    }
    else if (boxes_msg.encoding == "rgb8")
    {
        img = Mat(static_cast<int>(boxes_msg.height), static_cast<int>(boxes_msg.width), CV_8UC3);
        memcpy(img.data, boxes_msg.data.data(), boxes_msg.height * boxes_msg.width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        color_flag = 2;
    }
    else
    {
        sensor_msgs::Image img_;
        img_.header = boxes_msg.image_header;
        img_.height = boxes_msg.height;
        img_.width = boxes_msg.width;
        img_.is_bigendian = boxes_msg.is_bigendian;
        img_.step = boxes_msg.step;
        img_.data = boxes_msg.data;
        img_.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img_, sensor_msgs::image_encodings::MONO8); // 灰度图
        img = ptr->image;
        color_flag = 3;
    }
    // imshow("pic", img);
    // waitKey(1);
    
    // color_flag = 1; // 选择不同的图像预处理策略
    color_flag = 2; // 选择不同的图像预处理策略
    // cv::Mat candidate_box; // 感兴趣的区域
    // cv::Mat3d H;           // 单应性矩阵
    string class_of_box; // 类别
    vector<vector<double>> ellipse_output;
    std::vector<cv::Point> tri_output;
    std::vector<cv::Point2d> rect_output;
    // 标志计数
    int ellipse_num = 0;
    int tri_num = 0;
    int rect_num = 0;
    // 相似度
    double e_similarity = 0.0;
    double t_similarity = 0.0;
    double r_similarity = 0.0;
    // 编号
    double e_id = 0.0;
    double t_id = 0.0;
    double r_id = 0.0;
    // 内参
    // cv::Mat intrinsic = (Mat_<double>(3, 3) << 1765.682901, 0.000000, 782.352086, 0.000000, 1758.799034, 565.999397, 0, 0, 1);
    // cv::Mat distortion = (Mat_<double>(1, 4) << -0.060942, 0.058542, 0.001478, 0.002002);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << 0, 0.0, 0, 0);

    // yolov5_ros_msgs::BoundingBox内容：
    //  float64 probability
    //  int64 xmin
    //  int64 ymin
    //  int64 xmax
    //  int64 ymax
    //  int16 num
    //  string Class
    // yolov5_ros_msgs::BoundingBox box; // 一个框
    sensor_msgs::PointCloudPtr sign_points(new sensor_msgs::PointCloud);
    sign_points->header = boxes_msg.image_header;
    sign_points->header.frame_id = "sign";
    sensor_msgs::ChannelFloat32 sign_points_id;
    sensor_msgs::ChannelFloat32 u_of_sign_points0;
    sensor_msgs::ChannelFloat32 v_of_sign_points0; // 像素坐标(u,v)
    sensor_msgs::ChannelFloat32 u_of_sign_points1;
    sensor_msgs::ChannelFloat32 v_of_sign_points1;
    sensor_msgs::ChannelFloat32 u_of_sign_points2;
    sensor_msgs::ChannelFloat32 v_of_sign_points2;
    sensor_msgs::ChannelFloat32 u_of_sign_points3;
    sensor_msgs::ChannelFloat32 v_of_sign_points3;
    sensor_msgs::ChannelFloat32 u_of_sign_points4;
    sensor_msgs::ChannelFloat32 v_of_sign_points4;
    sensor_msgs::ChannelFloat32 sign_points_sim;

    for (yolov5_ros_msgs::BoundingBox box : boxes_msg.bounding_boxes)
    {
        // if ((box.ymax - box.ymin) < 40 && (box.xmax - box.xmin) < 64) // too small
        // {
        //     continue;
        //     // printf("\033[1;35m throw traffic sign !!!!\n    \033[0m");
        // }
        int xmin, xmax, ymin, ymax;
        judge(box.xmin, box.xmax, box.ymin, box.ymax, xmin, xmax, ymin, ymax);
        cv::Mat candidate_box = Mat::zeros(ymax - ymin, xmax - xmin, img.type());
        // cv::Mat candidate_box = img(Range(ymin, ymax), Range(xmin, xmax));
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                if ((i >= ymin && i < ymax) && (j >= xmin && j < xmax))
                {
                    candidate_box.at<Vec3b>(i - ymin, j - xmin) = img.at<Vec3b>(i, j);
                }
            }
        }
        // std::pair<string, cv::Mat> sign_temp;
        class_of_box = box.Class;
        char c = class_of_box[0];
        if (c == 'c')
        {
            if (e_xmax != 0)
            {
                std::vector<Point2f> rect1 = {Point2f(e_xmax, e_ymax), Point2f(e_xmax, e_ymin), Point2f(e_xmin, e_ymin), Point2f(e_xmin, e_ymax)};
                std::vector<Point2f> rect2 = {Point2f(box.xmax, box.ymax), Point2f(box.xmax, box.ymin), Point2f(box.xmin, box.ymin), Point2f(box.xmin, box.ymax)};
                if (rectanglesIntersect(rect1, rect2) && class_of_box != last_class_of_e)
                {
                    e_xmax = box.xmax;
                    e_xmin = box.xmin;
                    e_ymax = box.ymax;
                    e_ymin = box.ymin;
                    last_class_of_e = class_of_box;
                    continue;
                }
            }
            e_xmax = box.xmax;
            e_xmin = box.xmin;
            e_ymax = box.ymax;
            e_ymin = box.ymin;
            last_class_of_e = class_of_box;

            // a = ellipse_detect(candidate_box, color_flag, ellipse_output, e_similarity, xmin, ymin); // 输出一个中心点，长短轴和倾斜度，相当于五个double
            // ellipse_num++;
        }
        else if (c == 't')
        {
            if (t_xmax != 0)
            {
                std::vector<Point2f> rect1 = {Point2f(t_xmax, t_ymax), Point2f(t_xmax, t_ymin), Point2f(t_xmin, t_ymin), Point2f(t_xmin, t_ymax)};
                std::vector<Point2f> rect2 = {Point2f(box.xmax, box.ymax), Point2f(box.xmax, box.ymin), Point2f(box.xmin, box.ymin), Point2f(box.xmin, box.ymax)};
                if (rectanglesIntersect(rect1, rect2) && class_of_box != last_class_of_t)
                {
                    t_xmax = box.xmax;
                    t_xmin = box.xmin;
                    t_ymax = box.ymax;
                    t_ymin = box.ymin;
                    last_class_of_t = class_of_box;
                    continue;
                }
            }
            t_xmax = box.xmax;
            t_xmin = box.xmin;
            t_ymax = box.ymax;
            t_ymin = box.ymin;
            last_class_of_t = class_of_box;
            // a = tri_detect(candidate_box, color_flag, tri_output, t_similarity, xmin, ymin); // 输出四个点，相当于八个double
            // tri_num++;

            // vector<uchar> status;
            // vector<float> err;
            // static vector<cv::Point2f> t_cur_pts;
            // vector<cv::Point2f> t_forw_pts;
            // if (t_cur_pts.size() == 0 && t_sign.empty())
            // {
            //     cv::goodFeaturesToTrack(candidate_box, t_cur_pts, 15, 0.01, 5);
            //     sign_temp.first = class_of_box;
            //     sign_temp.second = candidate_box;
            //     t_sign.push_back(sign_temp);
            //     return;
            // }
            // // cv::calcOpticalFlowPyrLK(candidate_box, t_sign.back().second, t_cur_pts, t_forw_pts, status,
            // //                          err, cv::Size(7, 7), 2,
            // //                          cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            // //                          cv::OPTFLOW_USE_INITIAL_FLOW); // 与当前帧进行光流
            // // for (int i = 0; i < int(t_forw_pts.size()); i++)
            // //     if (status[i] && !inBorder(t_forw_pts[i]))
            // //         status[i] = 0;
            // // reduceVector(t_cur_pts, status);
            // // reduceVector(t_forw_pts, status);
            // if (t_forw_pts.size() < 5)
            // {
            //     tri_output.clear();
            //     break;
            // }
            // else
            // {
            //     cv::Mat show_img;
            //     cv::Mat tmp_img = candidate_box;
            //     cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);
            //     for (unsigned int j = 0; j < t_forw_pts.size(); j++)
            //     {
            //         cv::circle(tmp_img, t_forw_pts[j], 1, cv::Scalar(255, 0, 255), 2);
            //     }
            //     cv::imshow("track", tmp_img);
            //     cv::waitKey(5);
            //     t_cur_pts = t_forw_pts;
            //     tri_num++;
            //     sign_temp.first = class_of_box;
            //     sign_temp.second = candidate_box;
            //     t_sign.push_back(sign_temp);
            // }
            // while (t_sign.size() > 10)
            // {
            //     t_sign.pop_front();
            // }
        }
        else if (c == 'r')
        {
            if (r_xmax != 0)
            {
                std::vector<Point2f> rect1 = {Point2f(r_xmax, r_ymax), Point2f(r_xmax, r_ymin), Point2f(r_xmin, r_ymin), Point2f(r_xmin, r_ymax)};
                std::vector<Point2f> rect2 = {Point2f(box.xmax, box.ymax), Point2f(box.xmax, box.ymin), Point2f(box.xmin, box.ymin), Point2f(box.xmin, box.ymax)};
                if (rectanglesIntersect(rect1, rect2) && class_of_box != last_class_of_r)
                {
                    r_xmax = box.xmax;
                    r_xmin = box.xmin;
                    r_ymax = box.ymax;
                    r_ymin = box.ymin;
                    last_class_of_r = class_of_box;
                    continue;
                }
            }
            r_xmax = box.xmax;
            r_xmin = box.xmin;
            r_ymax = box.ymax;
            r_ymin = box.ymin;
            last_class_of_r = class_of_box;
            // 输出五个点，相当于十个double
            // if (rect_detect(candidate_box, color_flag, rect_output, r_similarity, xmin, ymin) == 1)
            // {
            //     rect_num++;
            // }
        }
        // cout << "rect_num" << rect_num << endl;
        /**********************************************************/
        // 计算中心点在相机系中的位置
        //  20221031_xjl
        if (ellipse_num > 0 && !ellipse_output.empty())
        {
            Point2f tmp_pts;
            std::vector<cv::Point2f> dis_pts, un_dis_pts;
            for (int i = 0; i < 1; i++)
            {
                tmp_pts.x = ellipse_output[i][0];
                tmp_pts.y = ellipse_output[i][1];
                // output.pop_back();
                dis_pts.push_back(tmp_pts);
            }
            // 只对中心做去畸变
            // cv::undistortPoints(dis_pts, un_dis_pts, intrinsic, distortion, cv::Mat(), intrinsic);
            un_dis_pts=dis_pts;
            // dis_pts.clear();
            // ROS_WARN("un_dis_pts : %f ,%f", un_dis_pts.back().x, un_dis_pts.back().y);
            // std::cout << "un_dis_pts" << un_dis_pts << std::endl;
            sensor_msgs::PointCloudPtr ellipse_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 u_of_point;   //
            sensor_msgs::ChannelFloat32 v_of_point;   // 像素坐标(u,v)
            sensor_msgs::ChannelFloat32 a_of_point;   // 长轴
            sensor_msgs::ChannelFloat32 b_of_point;   // 短轴
            sensor_msgs::ChannelFloat32 ori_of_point; // 倾斜度
            sensor_msgs::ChannelFloat32 sim;
            sensor_msgs::ChannelFloat32 id;
            cout << ellipse_num << endl;
            for (int i = 0; i < 1; i++)
            {
                pub_ellipse_count++; // 检测到椭圆的id（暂时没有用到）

                ellipse_points->header = boxes_msg.image_header;
                ellipse_points->header.frame_id = class_of_box;
                e_id = sign_class_to_id(class_of_box);
                geometry_msgs::Point32 p;

                // 20230825_xjl
                p.x = un_dis_pts[i].x;
                p.y = un_dis_pts[i].y;
                p.z = 1.0; // 矫正后归一化平面的3d点(x,y,1)

                ellipse_points->points.push_back(p);
                u_of_point.values.push_back(ellipse_output[i][0]);
                v_of_point.values.push_back(ellipse_output[i][1]);
                a_of_point.values.push_back(ellipse_output[i][2]);
                b_of_point.values.push_back(ellipse_output[i][3]);
                ori_of_point.values.push_back(ellipse_output[i][4]);
                sim.values.push_back(e_similarity);
                id.values.push_back(e_id);

                ellipse_points->channels.push_back(u_of_point);
                ellipse_points->channels.push_back(v_of_point);
                ellipse_points->channels.push_back(a_of_point);
                ellipse_points->channels.push_back(b_of_point);
                ellipse_points->channels.push_back(ori_of_point);
                ellipse_points->channels.push_back(sim);
                pub_ellipse.publish(ellipse_points);

                sign_points->points.push_back(p);
                sign_points_id.values.push_back(e_id);
                u_of_sign_points0.values.push_back(ellipse_output[i][0]);
                v_of_sign_points0.values.push_back(ellipse_output[i][1]);
                u_of_sign_points1.values.push_back(ellipse_output[i][2]);
                v_of_sign_points1.values.push_back(ellipse_output[i][3]);
                u_of_sign_points2.values.push_back(ellipse_output[i][4]);
                v_of_sign_points2.values.push_back(ellipse_output[i][0]);
                u_of_sign_points3.values.push_back(ellipse_output[i][1]);
                v_of_sign_points3.values.push_back(ellipse_output[i][2]);
                u_of_sign_points4.values.push_back(ellipse_output[i][3]);
                v_of_sign_points4.values.push_back(ellipse_output[i][4]);
                sign_points_sim.values.push_back(e_similarity);

                // sign_points->channels.push_back(id);
                // sign_points->channels.push_back(u_of_point);
                // sign_points->channels.push_back(v_of_point);
                // sign_points->channels.push_back(a_of_point);
                // sign_points->channels.push_back(b_of_point);
                // sign_points->channels.push_back(ori_of_point);
                // sign_points->channels.push_back(u_of_point);
                // sign_points->channels.push_back(v_of_point);
                // sign_points->channels.push_back(a_of_point);
                // sign_points->channels.push_back(b_of_point);
                // sign_points->channels.push_back(ori_of_point);
                // sign_points->channels.push_back(sim);
            }
            ellipse_num = ellipse_num - 1;
            ellipse_output.pop_back();
            dis_pts.clear();
            un_dis_pts.clear();
        }

        if (tri_num > 0 && !tri_output.empty())
        {
            std::vector<cv::Point2f> dis_pts, un_dis_pts;
            Point2f tmp_pts_center, tmp_pts_1, tmp_pts_2, tmp_pts_3;
            for (int i = 0; i < tri_num; i++)
            {
                tmp_pts_center.x = tri_output[4 * i].x;
                tmp_pts_center.y = tri_output[4 * i].y;
                tmp_pts_1.x = tri_output[4 * i + 1].x;
                tmp_pts_1.y = tri_output[4 * i + 1].y;
                tmp_pts_2.x = tri_output[4 * i + 2].x;
                tmp_pts_2.y = tri_output[4 * i + 2].y;
                tmp_pts_3.x = tri_output[4 * i + 3].x;
                tmp_pts_3.y = tri_output[4 * i + 3].y;
                // output.pop_back();
                dis_pts.push_back(tmp_pts_center);
                dis_pts.push_back(tmp_pts_1);
                dis_pts.push_back(tmp_pts_2);
                dis_pts.push_back(tmp_pts_3);
            }
            // 去畸变
            // cv::undistortPoints(dis_pts, un_dis_pts, intrinsic, distortion, cv::Mat(), intrinsic);
            un_dis_pts=dis_pts;
            // dis_pts.clear();
            // ROS_WARN("un_dis_pts : %f ,%f", un_dis_pts.back().x, un_dis_pts.back().y);
            // std::cout << "un_dis_pts" << un_dis_pts << std::endl;
            sensor_msgs::PointCloudPtr tri_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 u_of_center; //
            sensor_msgs::ChannelFloat32 v_of_center; // 像素坐标
            sensor_msgs::ChannelFloat32 u_of_point1; //
            sensor_msgs::ChannelFloat32 v_of_point1; // 像素坐标
            sensor_msgs::ChannelFloat32 u_of_point2; //
            sensor_msgs::ChannelFloat32 v_of_point2; // 像素坐标
            sensor_msgs::ChannelFloat32 u_of_point3; //
            sensor_msgs::ChannelFloat32 v_of_point3; // 像素坐标
            sensor_msgs::ChannelFloat32 sim;
            sensor_msgs::ChannelFloat32 id;
            for (int i = 0; i < tri_num; i++)
            {
                tri_points->header = boxes_msg.image_header;
                tri_points->header.frame_id = class_of_box;
                t_id = sign_class_to_id(class_of_box);

                geometry_msgs::Point32 p;
                p.x = un_dis_pts[4 * i].x;
                p.y = un_dis_pts[4 * i].y;
                p.z = 1; // 矫正后归一化平面的3d点(x,y,1)
                tri_points->points.push_back(p);
                u_of_center.values.push_back(un_dis_pts[4 * i].x);
                v_of_center.values.push_back(un_dis_pts[4 * i].y);
                u_of_point1.values.push_back(un_dis_pts[4 * i + 1].x);
                v_of_point1.values.push_back(un_dis_pts[4 * i + 1].y);
                u_of_point2.values.push_back(un_dis_pts[4 * i + 2].x);
                v_of_point2.values.push_back(un_dis_pts[4 * i + 2].y);
                u_of_point3.values.push_back(un_dis_pts[4 * i + 3].x);
                v_of_point3.values.push_back(un_dis_pts[4 * i + 3].y);
                sim.values.push_back(t_similarity);
                id.values.push_back(t_id);

                tri_points->channels.push_back(u_of_center);
                tri_points->channels.push_back(v_of_center);
                tri_points->channels.push_back(u_of_point1);
                tri_points->channels.push_back(v_of_point1);
                tri_points->channels.push_back(u_of_point2);
                tri_points->channels.push_back(v_of_point2);
                tri_points->channels.push_back(u_of_point3);
                tri_points->channels.push_back(v_of_point3);
                tri_points->channels.push_back(sim);
                pub_tri.publish(tri_points);

                sign_points->points.push_back(p);
                sign_points_id.values.push_back(t_id);
                u_of_sign_points0.values.push_back(un_dis_pts[4 * i].x);
                v_of_sign_points0.values.push_back(un_dis_pts[4 * i].y);
                u_of_sign_points1.values.push_back(un_dis_pts[4 * i + 1].x);
                v_of_sign_points1.values.push_back(un_dis_pts[4 * i + 1].y);
                u_of_sign_points2.values.push_back(un_dis_pts[4 * i + 2].x);
                v_of_sign_points2.values.push_back(un_dis_pts[4 * i + 2].y);
                u_of_sign_points3.values.push_back(un_dis_pts[4 * i + 3].x);
                v_of_sign_points3.values.push_back(un_dis_pts[4 * i + 3].y);
                u_of_sign_points4.values.push_back(un_dis_pts[4 * i + 3].x);
                v_of_sign_points4.values.push_back(un_dis_pts[4 * i + 3].y);
                sign_points_sim.values.push_back(t_similarity);
                // ROS_WARN("publish %f, at %f", tri_points->header.stamp.toSec(), ros::Time::now().toSec());
            }
            tri_num = 0;
            dis_pts.clear();
            un_dis_pts.clear();
        }

        if (rect_num > 0 && !rect_output.empty())
        {
            std::vector<cv::Point2d> dis_pts, un_dis_pts;
            Point2d tmp_pts_center, tmp_pts_1, tmp_pts_2, tmp_pts_3, tmp_pts_4;
            for (int i = 0; i < rect_num; i++)
            {
                tmp_pts_center = rect_output[5 * i];
                tmp_pts_1 = rect_output[5 * i + 1];
                tmp_pts_2 = rect_output[5 * i + 2];
                tmp_pts_3 = rect_output[5 * i + 3];
                tmp_pts_4 = rect_output[5 * i + 4];

                dis_pts.push_back(tmp_pts_center);
                dis_pts.push_back(tmp_pts_1);
                dis_pts.push_back(tmp_pts_2);
                dis_pts.push_back(tmp_pts_3);
                dis_pts.push_back(tmp_pts_4);
            }
            // 去畸变
            // cv::undistortPoints(dis_pts, un_dis_pts, intrinsic, distortion, cv::Mat(), intrinsic);
            un_dis_pts=dis_pts;
            sensor_msgs::PointCloudPtr rect_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 u_of_center;
            sensor_msgs::ChannelFloat32 v_of_center;
            sensor_msgs::ChannelFloat32 u_of_point1;
            sensor_msgs::ChannelFloat32 v_of_point1;
            sensor_msgs::ChannelFloat32 u_of_point2;
            sensor_msgs::ChannelFloat32 v_of_point2;
            sensor_msgs::ChannelFloat32 u_of_point3;
            sensor_msgs::ChannelFloat32 v_of_point3;
            sensor_msgs::ChannelFloat32 u_of_point4;
            sensor_msgs::ChannelFloat32 v_of_point4;
            sensor_msgs::ChannelFloat32 sim;
            sensor_msgs::ChannelFloat32 id;
            for (int i = 0; i < rect_num; i++)
            {
                // rect_points->header = boxes_msg.image_header;
                // rect_points->header.frame_id = class_of_box;
                r_id = sign_class_to_id(class_of_box);
                // cout << "r_id" << r_id << endl;
                geometry_msgs::Point32 p;
                p.x = un_dis_pts[5 * i].x;
                p.y = un_dis_pts[5 * i].y;
                p.z = 1; // 矫正后归一化平面的3d点(x,y,1)
                // rect_points->points.push_back(p); // 中心点
                // u_of_center.values.push_back(un_dis_pts[5 * i].x);
                // v_of_center.values.push_back(un_dis_pts[5 * i].y);
                // u_of_point1.values.push_back(un_dis_pts[5 * i + 1].x);
                // v_of_point1.values.push_back(un_dis_pts[5 * i + 1].y);
                // u_of_point2.values.push_back(un_dis_pts[5 * i + 2].x);
                // v_of_point2.values.push_back(un_dis_pts[5 * i + 2].y);
                // u_of_point3.values.push_back(un_dis_pts[5 * i + 3].x);
                // v_of_point3.values.push_back(un_dis_pts[5 * i + 3].y);
                // u_of_point4.values.push_back(un_dis_pts[5 * i + 4].x);
                // v_of_point4.values.push_back(un_dis_pts[5 * i + 4].y);
                // sim.values.push_back(r_similarity);
                // id.values.push_back(r_id);

                // rect_points->channels.push_back(u_of_center);
                // rect_points->channels.push_back(v_of_center);
                // rect_points->channels.push_back(u_of_point1);
                // rect_points->channels.push_back(v_of_point1);
                // rect_points->channels.push_back(u_of_point2);
                // rect_points->channels.push_back(v_of_point2);
                // rect_points->channels.push_back(u_of_point3);
                // rect_points->channels.push_back(v_of_point3);
                // rect_points->channels.push_back(u_of_point4);
                // rect_points->channels.push_back(v_of_point4);
                // rect_points->channels.push_back(sim);
                // pub_rect.publish(rect_points);

                sign_points->points.push_back(p);
                sign_points_id.values.push_back(r_id);
                u_of_sign_points0.values.push_back(un_dis_pts[5 * i].x);
                v_of_sign_points0.values.push_back(un_dis_pts[5 * i].y);
                u_of_sign_points1.values.push_back(un_dis_pts[5 * i + 1].x);
                v_of_sign_points1.values.push_back(un_dis_pts[5 * i + 1].y);
                u_of_sign_points2.values.push_back(un_dis_pts[5 * i + 2].x);
                v_of_sign_points2.values.push_back(un_dis_pts[5 * i + 2].y);
                u_of_sign_points3.values.push_back(un_dis_pts[5 * i + 3].x);
                v_of_sign_points3.values.push_back(un_dis_pts[5 * i + 3].y);
                u_of_sign_points4.values.push_back(un_dis_pts[5 * i + 4].x);
                v_of_sign_points4.values.push_back(un_dis_pts[5 * i + 4].y);
                sign_points_sim.values.push_back(r_similarity);
            }
            rect_num = 0;
            dis_pts.clear();
            un_dis_pts.clear();
        }
    }
    sign_points->channels.push_back(sign_points_id);
    sign_points->channels.push_back(u_of_sign_points0);
    sign_points->channels.push_back(v_of_sign_points0);
    sign_points->channels.push_back(u_of_sign_points1);
    sign_points->channels.push_back(v_of_sign_points1);
    sign_points->channels.push_back(u_of_sign_points2);
    sign_points->channels.push_back(v_of_sign_points2);
    sign_points->channels.push_back(u_of_sign_points3);
    sign_points->channels.push_back(v_of_sign_points3);
    sign_points->channels.push_back(u_of_sign_points4);
    sign_points->channels.push_back(v_of_sign_points4);
    sign_points->channels.push_back(sign_points_sim);
    if (!sign_points->points.empty())
    {
        pub_sign.publish(sign_points);
        // cout << "sign_points->points: " << sign_num << "pub points:" << sign_points->points.size() << endl;
    }

    /**********************************************************/
    // 20230519_xjl
    ellipse_output.clear();
    tri_output.clear();
    rect_output.clear();
    pre_image_time = cur_image_time;
    return;
    //         //     }
    //         // }
    //         // skip the first image; since no optical speed on frist image
    //         // if (!init_pub)
    //         // {
    //         //     init_pub = 1;
    //         // }
    //         // else

    // if (SHOW_ELLIPSE)
    // {
    //     ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
    //     //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
    //     cv::Mat stereo_img = ptr->image;
    //     cv::Mat show_img = ptr->image;

    // for (int i = 0; i < NUM_OF_CAM; i++)
    // {
    //     cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
    //     cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);

    // for (unsigned int j = 0; j < ellipse_num; j++)
    // {
    // double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
    // cv::ellipse( tmp_img, cv::Point2d(output[j][0],output[j][1]), cv::Size(output[j][2],output[j][3]),output[j][4], 0, 360, CV_RGB(0, 0, 255), 1, cv::LINE_AA, 0);
    // cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    //     //                 //draw speed line
    //     //                 /*
    //     //                 Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
    //     //                 Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
    //     //                 Vector3d tmp_prev_un_pts;
    //     //                 tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
    //     //                 tmp_prev_un_pts.z() = 1;
    //     //                 Vector2d tmp_prev_uv;
    //     //                 trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
    //     //                 cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
    //     //                 */
    // }
    // }
    //     pub_match.publish(ptr->toImageMsg());
    // }
    // }
}

void aruco_callback(const geometry_msgs::PoseStamped &aruco_pose)
{
    // ROS_INFO("processing camera %f",img_msg->header.stamp.toSec());

    return;
}

/*******************************************************/
// 20230211xjl
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double dt;
    if (first_imu)
    {
        pre_imu.time = imu_msg->header.stamp.toSec();
        pre_imu.dt = imu_datadt;
        dt = pre_imu.dt;
        pre_imu.dtheta.x() = imu_msg->angular_velocity.x * dt;
        pre_imu.dtheta.y() = imu_msg->angular_velocity.y * dt;
        pre_imu.dtheta.z() = imu_msg->angular_velocity.z * dt;
        pre_imu.dvel.x() = imu_msg->linear_acceleration.x * dt;
        pre_imu.dvel.y() = imu_msg->linear_acceleration.y * dt;
        pre_imu.dvel.z() = imu_msg->linear_acceleration.z * dt;
        pre_imu.odovel = 0;
        first_imu = 0;
        m_buf.lock();
        ins_window.push(pre_imu);
        m_buf.unlock();
        return;
    }
    if (imu_msg->header.stamp.toSec() <= pre_imu.time)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    cur_imu.time = imu_msg->header.stamp.toSec();
    cur_imu.dt = cur_imu.time - pre_imu.time;
    dt = cur_imu.dt;
    cur_imu.dtheta.x() = imu_msg->angular_velocity.x * dt;
    cur_imu.dtheta.y() = imu_msg->angular_velocity.y * dt;
    cur_imu.dtheta.z() = imu_msg->angular_velocity.z * dt;
    cur_imu.dvel.x() = imu_msg->linear_acceleration.x * dt;
    cur_imu.dvel.y() = imu_msg->linear_acceleration.y * dt;
    cur_imu.dvel.z() = imu_msg->linear_acceleration.z * dt;
    cur_imu.odovel = 0; // 轮速计？
    if (dt > imu_datadt * 1.5)
    {
        ROS_WARN("Lost IMU data!!!");
        long cnts = lround(dt / imu_datadt) - 1;
        IMU imudata = cur_imu;
        imudata.time = cur_imu.time - cur_imu.dt;
        m_buf.lock();
        while (cnts--)
        {
            imudata.time += imu_datadt;
            imudata.dt = imu_datadt;
            ins_window.push(imudata);

            cur_imu.dt = cur_imu.time - imudata.time;
            std::cout << "Append extra IMU data at " << imudata.time;
        }
    }
    ins_window.push(cur_imu);
    m_buf.unlock();
    pre_imu = cur_imu;

    return;
}
/*******************************************************/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); // 读取相机内参

    if (FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 20, img_callback);
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 1000, imu_callback, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber sub_ellipse_img = n.subscribe(IMAGE_TOPIC, 20, ellipse_detect_callback);
    // ros::Subscriber sub_tri_img = n.subscribe(IMAGE_TOPIC, 100, tri_detect_callback);
    // ros::Subscriber sub_rect_img = n.subscribe(IMAGE_TOPIC, 100, rect_detect_callback);
    //    The available fields are:
    //        probability,xmin,ymin,xmax,ymax,num,Class
    ros::Subscriber sub_detect_boxes = n.subscribe("/yolov5/BoundingBoxes", 50, boxes_callback);
    ros::Subscriber sub_pose = n.subscribe("/aruco_single/pose", 20, aruco_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 5000);
    // pub_ellipse = n.advertise<sensor_msgs::PointCloud>("ellipse", 20);
    // pub_tri = n.advertise<sensor_msgs::PointCloud>("tri", 20);
    // pub_rect = n.advertise<sensor_msgs::PointCloud>("rect", 20);

    pub_sign = n.advertise<sensor_msgs::PointCloud>("sign", 20);
    pub_camerainfo = n.advertise<sensor_msgs::CameraInfo>("left_camera_info", 10);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?