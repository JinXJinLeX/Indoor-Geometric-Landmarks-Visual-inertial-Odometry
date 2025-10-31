#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

#include <sensor_msgs/NavSatFix.h>
#include "common/angle.h"
#include "common/gpstime.h"
#include "common/types.h"
#include "common/rotation.h"
#include "common/earth.h"
#include "common/csv.h"
// #include "common/csv_myown.h"
#include <sstream>
#include <fstream>
#include <math.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// 20240306_xjl
#include "pmc/pmc_graph.h"
#include "pmc/pmc_utils.h"
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// #include "camodocal/camera_models/CameraFactory.h"
// #include "camodocal/camera_models/CataCamera.h"
// #include "camodocal/camera_models/PinholeCamera.h"
#define CSV_IO_NO_THREAD

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

// 20221022xjl
bool GINS_INTIALING = false;
bool VIO_INTIALING = false;
bool GVINS_INTIALING = false;
bool MAP_INTIALING = false;
std::queue<GNSS> gnss_buf;
std::queue<GNSS> map_buf;
std::queue<MARKER> aruco_buf;
std::queue<SIGN> sign_buf;
std::queue<pair<double, std::vector<SIGN>>> sign_queue;

std::vector<IMU> ins_window;
double ZERO_VELOCITY_GYR_THRESHOLD = 0.02;
double ZERO_VELOCITY_ACC_THRESHOLD = 0.1;
Eigen::Vector3d origin;
std::queue<sensor_msgs::PointCloudConstPtr> ellipse_buf;
std::queue<sensor_msgs::PointCloudConstPtr> tri_buf;
std::queue<sensor_msgs::PointCloudConstPtr> rect_buf;
std::queue<double> timelist;
cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 952.011, 0.000000, 661.733, 0.000000, 959.650, 375.598, 0, 0, 1);
// cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 892.982544, 0.000000, 611.368633, 0.000000, 900.910278, 353.958833, 0, 0, 1);
cv::Mat distortion;
// cv::Mat distortion = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0,0);
typedef struct StateData
{
    double time;

    double pose[7]; // pose : 3 + 4 = 7

    // mix parameters
    // vel + bias : 3 + 6 = 9
    // vel + bias + sodo : 3 + 6 + 1 = 10
    // vel + bias + sodo + abv : 3 + 6 + 1 + 2 = 12
    // vel + bias + scale : 3 + 6 + 6 = 15
    // vel + bias + sodo + scale : 3 + 6 + 1 + 6 = 16
    // vel + bias + sodo + scale + abv : 3 + 6 + 1 + 6 + 2 = 18
    // double mix[18];
} StateData;

std::queue<StateData> statelist;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g; // a_0 = q * (a0 - ba) -g

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; // w_0 = 0.5 * (w0 + w) - bg
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);                       // w * t

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g; // a_1 = q * (a1 - ba) -g

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // a = 0.5 * (a_0+a_1)

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

/**********************************************************/
// 20221022xjl
bool detectZeroVelocity(std::vector<IMU> &imu_buffer, double imudatarate, vector<double> &average)
{
    auto size = (double)(imu_buffer.size());
    double size_invert = 1.0 / size;

    double sum[6];
    double std[6];

    average.resize(6);
    average[0] = average[1] = average[2] = average[3] = average[4] = average[5] = 0;
    for (auto &imu : imu_buffer)
    {
        average[0] += imu.dtheta[0];
        average[1] += imu.dtheta[1];
        average[2] += imu.dtheta[2];
        average[3] += imu.dvel[0];
        average[4] += imu.dvel[1];
        average[5] += imu.dvel[2];

        // average[0] += imu->angular_velocity.x * 0.01;
        // average[1] += imu->angular_velocity.y * 0.01;
        // average[2] += imu->angular_velocity.z * 0.01;
        // average[3] += imu->linear_acceleration.x* 0.01;
        // average[4] += imu->linear_acceleration.y* 0.01;
        // average[5] += imu->linear_acceleration.z* 0.01;
    }

    average[0] *= size_invert;
    average[1] *= size_invert;
    average[2] *= size_invert;
    average[3] *= size_invert;
    average[4] *= size_invert;
    average[5] *= size_invert;
    printf("average:%lf  %lf    %lf     %lf     %lf    %lf      \n", average[0], average[1], average[2], average[3], average[4], average[5]);

    sum[0] = sum[1] = sum[2] = sum[3] = sum[4] = sum[5] = 0;
    for (auto &imu : imu_buffer)
    {
        sum[0] += (imu.dtheta[0] - average[0]) * (imu.dtheta[0] - average[0]);
        sum[1] += (imu.dtheta[1] - average[1]) * (imu.dtheta[1] - average[1]);
        sum[2] += (imu.dtheta[2] - average[2]) * (imu.dtheta[2] - average[2]);
        sum[3] += (imu.dvel[0] - average[3]) * (imu.dvel[0] - average[3]);
        sum[4] += (imu.dvel[1] - average[4]) * (imu.dvel[1] - average[4]);
        sum[5] += (imu.dvel[2] - average[5]) * (imu.dvel[2] - average[5]);

        //     sum[0] += (imu->angular_velocity.x * 0.01- average[0]) * (imu->angular_velocity.x * 0.01- average[0]);
        //     sum[1] += (imu->angular_velocity.y * 0.01- average[1]) * (imu->angular_velocity.y * 0.01- average[1]);
        //     sum[2] += (imu->angular_velocity.z * 0.01- average[2]) * (imu->angular_velocity.z * 0.01- average[2]);
        //     sum[3] += (imu->linear_acceleration.x * 0.01- average[3]) * (imu->linear_acceleration.x * 0.01- average[3]);
        //     sum[4] += (imu->linear_acceleration.y * 0.01- average[4]) * (imu->linear_acceleration.y * 0.01- average[4]);
        //     sum[5] += (imu->linear_acceleration.z * 0.01- average[5]) * (imu->linear_acceleration.z * 0.01- average[5]);
    }

    // 速率形式
    std[0] = sqrt(sum[0] * size_invert) * imudatarate;
    std[1] = sqrt(sum[1] * size_invert) * imudatarate;
    std[2] = sqrt(sum[2] * size_invert) * imudatarate;
    std[3] = sqrt(sum[3] * size_invert) * imudatarate;
    std[4] = sqrt(sum[4] * size_invert) * imudatarate;
    std[5] = sqrt(sum[5] * size_invert) * imudatarate;
    printf("velocity:%lf  %lf    %lf     %lf     %lf    %lf      \n", std[0], std[1], std[2], std[3], std[4], std[5]);

    if ((std[0] < ZERO_VELOCITY_GYR_THRESHOLD) && (std[1] < ZERO_VELOCITY_GYR_THRESHOLD) &&
        (std[2] < ZERO_VELOCITY_GYR_THRESHOLD) && (std[3] < ZERO_VELOCITY_ACC_THRESHOLD) &&
        (std[4] < ZERO_VELOCITY_ACC_THRESHOLD) && (std[5] < ZERO_VELOCITY_ACC_THRESHOLD))
    {

        return true;
    }
    return false;
}

// 20221022xjl
bool gvinsInitialization()
{
    // queue<sensor_msgs::ImuConstPtr> imu_buf,std::queue<GNSS> &gnss_buf
    // queue<sensor_msgs::ImuConstPtr> imu_buf,std::queue<GNSS> &gnss_buf
    static GNSS gnss, last_gnss;
    static bool flag = true;
    if (flag == true)
    {
        last_gnss = gnss;
        gnss = gnss_buf.front();
    }

    double time;
    // deque<sensor_msgs::ImuConstPtr> imu_windows;
    // sensor_msgs::ImuConstPtr imu;
    static IMU imu, last_imu;
    std::vector<IMU> imu_windows;
    // 是否有两帧gnss
    if ((gnss.time == 0) || (last_gnss.time == 0))
    {
        std::cout << "TIME==0" << std::endl;
        ;
        gnss_buf.pop();
        return false;
    }
    // if (imu_buf.empty())
    //     return false;
    if (!(imu_buf.back()->header.stamp.toSec() > gnss.time))
    {
        flag = false;
        // ROS_WARN("wait for imu, only should happen at the beginning");
        return false;
    }
    flag = true;
    if (!(imu_buf.front()->header.stamp.toSec() < gnss.time))
    {
        gnss_buf.pop();
        ROS_WARN("throw gnss, only should happen at the beginning");
        return false;
    }
    // 剩下的情况就是
    //  imu                  ***********
    //  gnss                        *    *    *
    // 当imu的时间戳在gnss时间戳之间就塞入队列中
    //  time = gnss.time-last_gnss.time;
    while (imu_buf.front()->header.stamp.toSec() < gnss.time && !imu_buf.empty())
    {
        if (imu_buf.front()->header.stamp.toSec() < (last_gnss.time))
        {
            imu_buf.pop();
            continue;
        }
        imu.time = imu_buf.front()->header.stamp.toSec();
        if (imu.time - last_gnss.time <= (1 / IMUDATARATE))
            imu.dt = imu.time - last_gnss.time;
        imu.dt = imu.time - last_imu.time;
        // printf("gnss_dt:%d",gnss.time);
        // std::cout<<"gnss_dt:"<<gnss.time<<std::endl;;

        // std::cout<<"gnss_dt:"<<time<<std::endl;;

        // std::cout<<"dt:"<<imu.dt<<std::endl;;
        imu.dtheta[0] = imu_buf.front()->angular_velocity.x * imu.dt;
        imu.dtheta[1] = imu_buf.front()->angular_velocity.y * imu.dt;
        imu.dtheta[2] = imu_buf.front()->angular_velocity.z * imu.dt;
        imu.dvel[0] = imu_buf.front()->linear_acceleration.x * imu.dt;
        imu.dvel[1] = imu_buf.front()->linear_acceleration.y * imu.dt;
        imu.dvel[2] = imu_buf.front()->linear_acceleration.z * imu.dt;
        imu_windows.emplace_back(imu);
        last_imu = imu;
        imu_buf.pop();
    }
    // 再塞一帧
    if (!imu_buf.empty())
    {
        imu.time = imu_buf.front()->header.stamp.toSec();
        imu.dtheta[0] = imu_buf.front()->angular_velocity.x * imu.dt;
        imu.dtheta[1] = imu_buf.front()->angular_velocity.y * imu.dt;
        imu.dtheta[2] = imu_buf.front()->angular_velocity.z * imu.dt;
        imu.dvel[0] = imu_buf.front()->linear_acceleration.x * imu.dt;
        imu.dvel[1] = imu_buf.front()->linear_acceleration.y * imu.dt;
        imu.dvel[2] = imu_buf.front()->linear_acceleration.z * imu.dt;
        imu_windows.emplace_back(imu);
    }

    gnss_buf.pop();
    std_msgs::Header head = imu_buf.front()->header;

    if (imu_windows.size() < 20)
    {
        std::cout << "IMU BUFFER  IS  EMPTY" << std::endl;
        ;
        return false;
    }

    // 形如
    //  imu         *********
    //  gnss                    *
    //  if (imu_buf.front()->header.stamp.toSec() <= gnss.time)
    //  {
    //      for (int i = 0; i < imu_buf.size(); i++)
    //      {
    //          imu=imu_buf.front();
    //          imu_buf.pop();
    //          imu_windows.emplace_back(imu);
    //      }
    //  }
    //  if (imu_windows.size() < 20) {
    //      std::cout<<"IMU BUFFER  IS  EMPTY"<<std::endl;;
    //      // imu_windows.clear();
    //      return false;
    //  }

    vector<double> average;
    static Vector3d bg{0, 0, 0};
    static Vector3d initatt{0, 0, 0};
    // 零速检测查看p、q的方差/频率
    bool is_zero_velocity = detectZeroVelocity(imu_windows, IMUDATARATE, average);
    estimator.ini_R.Identity();
    std::cout << "imu_window_size:" << imu_windows.size() << "is_zero_velocity" << is_zero_velocity << "imu_buf_size" << imu_buf.size() << std::endl;
    ;

    imu_windows.clear();

    if (is_zero_velocity)
    {
        // 正好给出陀螺零偏的估计值
        bg = Vector3d(average[0], average[1], average[2]);
        bg *= IMUDATARATE;

        // 重力调平获取横滚俯仰角
        Vector3d fb(average[3], average[4], average[5]);
        fb *= IMUDATARATE;

        // G是文件里给的重力常量(0,0,9.81007)
        initatt[0] = -asin(fb[1] / G.z());
        initatt[1] = asin(fb[0] / G.z());
        // initatt[0] = 0;
        // initatt[1] = 0;

        std::cout << "Zero velocity get gyroscope bias " << bg.transpose() * 3600 * R2D << ", roll " << initatt[0] * R2D
                  << ", pitch " << initatt[1] * R2D << std::endl;
        ;
        return false; // 零速检测没能初始化yaw角
    }
    Vector3d velocity = Vector3d::Zero();
    // 若不是零速
    if (!is_zero_velocity)
    {
        // 一般不可观
        if (gnss.isyawvalid)
        {
            initatt[2] = last_gnss.yaw;
            std::cout << "Initialized heading from dual-antenna GNSS as " << initatt[2] * R2D << " deg" << std::endl;
            ;
        }
        else
        {
            velocity = gnss.blh - last_gnss.blh; // 此时已经是东北天坐标系下的坐标
            if (velocity.norm() < 0.2)
            {
                return false;
            }
            std::cout << "velocity" << velocity << std::endl;
            ;
            // 这里的MINMUM_ALIGN_VELOCITY设小一点可以快点初始化
            // if (velocity.norm() < MINMUM_ALIGN_VELOCITY)
            // 求出yaw角，求法是反正切，相当于是求得(x,y)与x轴的夹角，单位是弧度
            //  initatt[2] = atan2(velocity[1], velocity[0]);
            // 此处的前右下和右前上写法一样?

            // 20221117xjl
            Vector3d fb(average[3], average[4], average[5]);
            fb *= IMUDATARATE;
            // initatt[0] = -asin(fb[1] / G.z());
            // initatt[1] = asin(fb[0] / G.z());
            // initatt[0] = 0;
            // initatt[1] = 0;

            // initatt[0] =atan(velocity.z() / sqrt(velocity.x() * velocity.x() + velocity.y() * velocity.y()));
            initatt[0] = 0;
            initatt[1] = 0;
            initatt[2] = atan2(velocity.x(), velocity.y());

            // double yaw;
            // if (velocity[0] > 0 && velocity[1] > 0)
            // {
            //     yaw = -(90 * D2R - abs(atan2(velocity[1], velocity[0])));
            // }
            // if (velocity.x() < 0 && velocity.y() > 0)
            // {
            //     yaw = 90 * D2R - abs(atan2(velocity.y(), velocity.x()));
            // }
            // if (velocity.x() < 0 && velocity.y() < 0)
            // {
            //     yaw = 90 * D2R + abs(atan2(velocity.y(), velocity.x()));
            // }
            // if (velocity.x() > 0 && velocity.y() < 0)
            // {
            //     yaw = -(90 * D2R + abs(atan2(velocity.y(), velocity.x())));
            // }
            // initatt[2] = yaw ;
            std::cout << "Initialized heading from GNSS positioning as " << initatt[2] * R2D << " deg" << std::endl;
            ;
        }
    }
    Matrix3d RRR;
    RRR << 1, 0, 0, 0, 0, -1, 0, 1, 0;
    RRR = RRR * Rotation::euler2quaternion(initatt);
    std::cout << initatt * R2D << std::endl;

    // estimator.ini_P = (last_gnss.blh - Rotation::euler2quaternion(initatt) * ANTLEVER ); // 载体位置

    estimator.ini_P = (last_gnss.blh - RRR.transpose() * ANTLEVER); // 载体位置

    // estimator.ini_P = (last_gnss.blh );//载体位置
    // estimator.ini_Q = Quaterniond(Rotation::euler2quaternion(initatt)); // 姿态
    estimator.ini_Q = Quaterniond(RRR); // 姿态
    estimator.ini_Q.normalize();
    estimator.ini_R = estimator.ini_Q.toRotationMatrix();
    // pubLatestOdometry(tmp_P, tmp_Q, tmp_V, head);
    ROS_INFO("GVNIS_INITIALED!!!");
    std::cout << "estimator.ini_R : " << estimator.ini_R << std::endl;
    std::cout << "estimator.ini_P : " << estimator.ini_P << std::endl;
    return true;
}

bool same_signin_VectorMap(string str)
{
    io::CSVReader<10, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '\"'>> in("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv");
    in.read_header(io::ignore_missing_column, "ID", "class", "p_x", "p_y", "p_z", "n_x", "n_y", "n_z", "size", "time");
    string classofsign;
    double x, y, z, nx, ny, nz, size, t;
    int ID;
    while (in.read_row(ID, classofsign, x, y, z, nx, ny, nz, size, t))
    {
        if (classofsign == str)
        {
            return true;
        }
    }
}

// 20230607_xjl
// 输入：坐标、类别
// 输出：地图csv文件中的对应标志
bool parseVectorMap(int &id, Vector3d p, string str, vector<pair<pair<Vector3d, Vector3d>, double>> &p_list, double time, int &id_)
{
    p_list.clear();
    // io::CSVReader<7> in("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map1.csv");
    // in.read_header(io::ignore_extra_column, "class", "p_x", "p_y", "p_z", "yaw", "size", "time");

    io::CSVReader<10, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '\"'>> in("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv");
    in.read_header(io::ignore_missing_column, "ID", "class", "p_x", "p_y", "p_z", "n_x", "n_y", "n_z", "size", "time");
    // in.read_header(io::ignore_missing_column, "class", "p_x", "p_y", "p_z", "yaw", "size");
    string classofsign;
    double x, y, z, nx, ny, nz, size, t;
    double res;
    int ID;
    // vector<double> px,py,pz,qx,qy,qz,qw;
    // pair<pair<cv::Point3d, double>, double> p_;
    pair<pair<Vector3d, Vector3d>, double> p_;
    // while (in.read_row(classofsign, x, y, z, yaw, size, t))
    while (in.read_row(ID, classofsign, x, y, z, nx, ny, nz, size, t))
    {
        ++id_;
        // if (classofsign.find(str))
        if (classofsign == str)
        {
            res = ((x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1]) + (z - p[2]) * (z - p[2]));
            if (res <= 9) // 误差在5米内
            {
                printf("find it in csv!\n");
                p_.first.first.x() = x;
                p_.first.first.y() = y;
                p_.first.first.z() = z;
                p_.first.second.x() = nx;
                p_.first.second.y() = ny;
                p_.first.second.z() = nz;
                p_.second = size;
                p_list.push_back(p_);
                std::cout << "res" << res << std::endl;
                ;
                id = ID;
                break;
            }
        }
    }
    if (p_list.empty())
    {
        printf("don't find it in csv!\n");
        return false;
    }
    return true;
}

bool InitialVectorMap(Estimator estimator_)
{
    io::CSVReader<10, io::trim_chars<' ', '\t'>, io::double_quote_escape<',', '\"'>> in("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv");
    in.read_header(io::ignore_missing_column, "ID", "class", "p_x", "p_y", "p_z", "n_x", "n_y", "n_z", "size", "time");
    string classofsign;
    double x, y, z, nx, ny, nz, size, t;
    int ID;
    int is_sign_find = -1;
    vector<Vector2d> uv;
    int i = 0;
    while (in.read_row(ID, classofsign, x, y, z, nx, ny, nz, size, t))
    {
        i++;
        estimator_.para_sign_Pose[i][0] = x;
        estimator_.para_sign_Pose[i][1] = y;
        estimator_.para_sign_Pose[i][2] = z;
        estimator_.para_sign_Pose[i][3] = nx;
        estimator_.para_sign_Pose[i][4] = ny;
        estimator_.para_sign_Pose[i][5] = nz;
        estimator_.map_manager.initialSign(ID, classofsign, {x, y, z}, {nx, ny, nz}, t, uv, is_sign_find);
        SIGN tempsign;
        Matrix3d RR_;
        tempsign.signclass = classofsign;
        tempsign.C = Vector3d{x, y, z};
        tempsign.cvPoints.clear();
        tempsign.ric = estimator.ric[0];
        tempsign.tic = estimator.tic[0];
        tempsign.N = Vector3d{nx, ny, nz};
        RR_.block<1, 3>(0, 0) << ((Vector3d{0, 0, -1}).cross(tempsign.N)).x(), ((Vector3d{0, 0, -1}).cross(tempsign.N)).y(), ((Vector3d{0, 0, -1}).cross(tempsign.N)).z();
        RR_.block<1, 3>(2, 0) << nx, ny, nz;
        RR_.block<1, 3>(1, 0) << 0, 0, -1;
        tempsign.q = Quaterniond(RR_.transpose());
        tempsign.q.normalize();
        tempsign.scale = size;
        estimator.mapforsign.push_back(tempsign);
    }
    int j = 0;
    for (auto sign : estimator_.map_manager.sign)
    {
        estimator_.para_sign_Pose[j][0] = sign.C_.x();
        estimator_.para_sign_Pose[j][1] = sign.C_.y();
        estimator_.para_sign_Pose[j][2] = sign.C_.z();
        estimator_.para_sign_Pose[j][3] = sign.C_.x();
        estimator_.para_sign_Pose[j][4] = sign.C_.y();
        estimator_.para_sign_Pose[j][5] = sign.C_.z();
        j++;
    }
    return true;
}

// 重力投影
bool find_gravity(int index, Vector3d &g)
{
    // double dt;
    // Matrix3d R; // 姿态R
    // Vector3d P; // 位置P
    // Matrix3d r; // 外参r
    // Vector3d t; // 外参t

    // Vector3d g; // 相机系下的重力g
    if (index == -1)
        return false;
    // double time = estimator.Headers[index].stamp.toSec();
    // R = estimator.Rs[index];
    // P = estimator.Ps[index];
    // r = estimator.ric[0];
    // t = estimator.tic[0];

    g = estimator.ric[0].inverse() * estimator.Rs[index].inverse() * (-estimator.g);
    // g = estimator.ric[0].transpose() * estimator.g;
    return true;
}

Eigen::Matrix3d eulerAnglesToRotationMatrix(Vector3d &theta)
{

    Eigen::Matrix3d R_x, R_y, R_z;
    R_x << 1, 0, 0,
        0, cos(theta[0]), -sin(theta[0]),
        0, sin(theta[0]), cos(theta[0]);
    // Calculate rotation about y axis
    R_y << cos(theta[1]), 0, sin(theta[1]),
        0, 1, 0,
        -sin(theta[1]), 0, cos(theta[1]);
    // Calculate rotation about z axis
    R_z << cos(theta[2]), -sin(theta[2]), 0,
        sin(theta[2]), cos(theta[2]), 0,
        0, 0, 1;
    // Combined rotation matrix
    Eigen::Matrix3d R = R_z * R_y * R_x;
    return R;
}

void filterHomographyDecompByVisibleRefpoints(cv::InputArrayOfArrays _rotations,
                                              cv::InputArrayOfArrays _normals,
                                              cv::InputArray _beforeRectifiedPoints,
                                              cv::InputArray _afterRectifiedPoints,
                                              std::vector<int> &possibleSolutions,
                                              cv::InputArray _pointsMask)
{
    // CV_Assert(_beforeRectifiedPoints.type() == CV_32FC2 && _afterRectifiedPoints.type() == CV_32FC2);
    // CV_Assert(_pointsMask.empty() || _pointsMask.type() == CV_8U);

    cv::Mat beforeRectifiedPoints = _beforeRectifiedPoints.getMat();
    cv::Mat afterRectifiedPoints = _afterRectifiedPoints.getMat();
    cv::Mat pointsMask = _pointsMask.getMat();
    int nsolutions = (int)_rotations.total();
    int npoints = (int)beforeRectifiedPoints.total();
    // CV_Assert(pointsMask.empty() || pointsMask.checkVector(1, CV_8U) == npoints);
    const uchar *pointsMaskPtr = pointsMask.data;

    std::vector<uchar> solutionMask(nsolutions, (uchar)1);
    std::vector<cv::Mat> normals(nsolutions);
    std::vector<cv::Mat> rotnorm(nsolutions);
    cv::Mat R;

    for (int i = 0; i < nsolutions; i++)
    {
        _normals.getMat(i).convertTo(normals[i], CV_64F);
        // CV_Assert(normals[i].total() == 3);
        _rotations.getMat(i).convertTo(R, CV_64F);
        rotnorm[i] = R * normals[i];
        // CV_Assert(rotnorm[i].total() == 3);
    }

    for (int j = 0; j < npoints; j++)
    {
        if (!pointsMaskPtr || pointsMaskPtr[j])
        {
            cv::Point2d prevPoint = beforeRectifiedPoints.at<cv::Point2d>(j);
            cv::Point2d currPoint = afterRectifiedPoints.at<cv::Point2d>(j);

            for (int i = 0; i < nsolutions; i++)
            {
                if (!solutionMask[i])
                    continue;

                const double *normal_i = normals[i].ptr<double>();
                const double *rotnorm_i = rotnorm[i].ptr<double>();
                double prevNormDot = normal_i[0] * prevPoint.x + normal_i[1] * prevPoint.y + normal_i[2];
                double currNormDot = rotnorm_i[0] * currPoint.x + rotnorm_i[1] * currPoint.y + rotnorm_i[2];

                if (prevNormDot <= 0 || currNormDot <= 0)
                {
                    solutionMask[i] = (uchar)0;
                }
            }
        }
    }

    for (int i = 0; i < nsolutions; i++)
        if (solutionMask[i])
        {
            possibleSolutions.push_back(1);
            // std::cout << "suc" << std::endl;;
        }
        else
        {
            possibleSolutions.push_back(0);
            // std::cout << "fail" << std::endl;;
        }

    return;
}

// 20240522_xjl
bool process_Ellipse(SIGN &sign_e, cv::Mat intrinsic, cv::Mat distortion, int k)
{
    queue<sensor_msgs::PointCloudConstPtr> ellipse_buf;
    Vector3d ep;
    Vector3d np;
    double time;
    Eigen::Matrix3d RR_;
    Eigen::Vector3d PP_;
    vector<pair<pair<Eigen::Vector3d, Eigen::Vector3d>, double>> p_list;
    double u, v;
    double a, b;
    double ori;
    double ax, ay, bx, by;
    double m, n;
    double similarty;
    double dis;                   // 相机光心与标志中心距离
    double true_ellipse_d = 0.15; // 真实半径15cm
    string classofsign;
    Matrix3d intrinsic_;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    Vector3d p;
    double fx, fy, cx, cy, k1, k2, p1, p2;
    double in;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);
    time = sign_e.time;
    classofsign = sign_e.signclass;
    p.x() = sign_e.C.x(); // 去畸变后的坐标
    p.y() = sign_e.C.y();
    p.z() = sign_e.C.z();   // z=1，归一化
    u = sign_e.features[0]; // u,v
    v = sign_e.features[1];
    a = sign_e.features[2]; // a,b
    b = sign_e.features[3];
    ori = sign_e.features[4]; // ori
    similarty = sign_e.similarty;
    ax = u + a * cos(ori * M_PI / 180.0);
    ay = v + a * sin(ori * M_PI / 180.0);
    bx = u - a * cos(ori * M_PI / 180.0);
    by = v - a * sin(ori * M_PI / 180.0);
    // 求深度
    in = sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy) + (fx + fy) * (fx + fy) / 4);
    m = sqrt((fx + fy) * (fx + fy) / 4 + (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy));
    n = sqrt((fx + fy) * (fx + fy) / 4 + (bx - cx) * (bx - cx) + (by - cy) * (by - cy));
    dis = true_ellipse_d / tan(0.5 * acos((m * m + n * n - 4 * a * a) / (2 * m * n)));
    dis = 1 / dis * 0.2 * true_ellipse_d + dis;
    double nn = dis / in;
    ep.x() = (u - cx) * nn;
    ep.y() = (v - cy) * nn;
    ep.z() = (fx + fy) / 2 * nn;
    np = estimator.Rs[k] * (estimator.ric[0] * ep + estimator.tic[0]) + estimator.Ps[k]; // 得到标志中心在世界坐标系下坐标
    int index = k;                                                                       // 标志对应的滑窗位置索引

    // csv读文件
    // 找到重力对应的投影
    Vector3d g;
    vector<Vector2d> uv;
    Vector3d NN;
    Vector3d CC = np;
    std::vector<Vector3d> ps, pw; // 相机系下的坐标,世界系下的坐标
    std::vector<cv::Point2d> pc;  // 像素坐标（深度为焦距）
    if (!find_gravity(index, g))  // 找到图像对应时间戳下重力g转为相机系下的g
    {
        return false;
    }
    g.normalize();
    // 输入：在相机系下的重力g
    // 标志中心像素坐标p
    // 标志中心点相机系下位置ep
    // 长短轴、倾斜角以及
    // 得到相机平面的标志最高点、最低点、最远点、最近点(其中以靠左边的这个点，对应标志靠右的点为x轴)
    estimator.computeEllipseLineIntersection(g, cv::Point2d(u, v), ep, a, b, ori, pc, ps, pw, intrinsic, index, dis, NN); // g投影到相机平面建立坐标系，与椭圆相交并且得出xy轴的交点(这是一个数学过程)
    int id = -1;
    int id_ = 0;
    int is_sign_find = 0;
    estimator.map_manager.addSignCheck(np, classofsign, id); // 在管理器中寻找是否有同一个标志
    for (int i = 0; i < pc.size(); i++)
    {
        uv.push_back({(pc[i].x - cx) / fx, (pc[i].y - cy) / fy});
    }
    if (id != -1) // 若在管理器中找到了该标志
    {
        printf("find it in map_manager!\n");
        is_sign_find = 1;
        if (!estimator.map_manager.sign.empty())
        {
            for (auto it = estimator.map_manager.sign.begin(), it_next = estimator.map_manager.sign.begin(); it != estimator.map_manager.sign.end(); it = it_next) // 遍历滑窗标志
            {
                if (it->classify == classofsign && ((it->C_ - np).norm() < 2) && id == it->sign_id)
                {
                    estimator.map_manager.initialSign(id, classofsign, it->C_, it->N_, time, uv, is_sign_find);
                    NN = it->N_;
                    NN.normalize();
                    CC = it->C_;
                    break;
                }
                it_next++;
            }
        }
    }
    else // 管理器中没有，去csv文件中找
    {
        if (parseVectorMap(id, np, classofsign, p_list, time, id_)) // 使用id读地图得到所需要的标志的先验信息p_list
        {
            // 此时的id不为-1
            if (!p_list.empty())
            {
                is_sign_find = 2;
                NN = p_list.front().first.second;
                NN.normalize();
                CC = p_list.front().first.first;
                estimator.map_manager.initialSign(id, classofsign, p_list.front().first.first, NN, time, uv, is_sign_find); // 初始化路标到管理器中，只负责加入，和add features一样
            }
        }
    }
    printf("\033[1;31m dis of ellipse:%f\nc_pos_ellipse: %f, %f, %f \nn_pos_ellipse:%f,%f,%f\n\033[0m",
           dis, ep.x(), ep.y(), ep.z(), np.x(), np.y(), np.z());
    std::vector<cv::Point3d> ps_0;
    ps_0.push_back({0, 0, 0.01});     // 路标坐标系下的关键点
    ps_0.push_back({0, 0.15, 0.01});  // 圆下
    ps_0.push_back({-0.15, 0, 0.01}); // 圆右
    ps_0.push_back({0, -0.15, 0.01}); // 圆上
    ps_0.push_back({0.15, 0, 0.01});  // 圆左

    ps_0.push_back({-0.15, -0.15, 0.01}); // 圆右上
    ps_0.push_back({-0.15, 0.15, 0.01});  // 圆右下
    ps_0.push_back({0.15, 0.15, 0.01});   // 圆左下
    ps_0.push_back({0.15, -0.15, 0.01});  // 圆左上

    // ps_0.push_back({0, 0, 0.16});  // 圆前
    cv::Mat rvec(3, 1, CV_64FC1);
    cv::Mat tvec(3, 1, CV_64FC1);
    cv::Mat rot;
    // cv::solvePnP(ps_0, pc, intrinsic, distortion, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    cv::solvePnPRansac(ps_0, pc,
                       intrinsic, distortion,
                       rvec, tvec, false, 200, 5.0, 0.99);
    /*****************draw key points of sign******************/
    vector<cv::Point2d> draw;
    if (is_sign_find != 0)
    {
        vector<Vector3d> worldpoint;
        worldpoint.push_back(CC);
        worldpoint.push_back(CC - 0.15 * G / G.norm());                       // 圆下
        worldpoint.push_back(CC - 0.15 * NN.cross(G) / (NN.cross(G)).norm()); // 圆右
        worldpoint.push_back(CC + 0.15 * G / G.norm());                       // 圆上
        worldpoint.push_back(CC + 0.15 * NN.cross(G) / (NN.cross(G)).norm()); // 圆左

        worldpoint.push_back(CC - 0.15 * NN.cross(G) / (NN.cross(G)).norm() + 0.15 * G / G.norm()); // 圆右上
        worldpoint.push_back(CC - 0.15 * NN.cross(G) / (NN.cross(G)).norm() - 0.15 * G / G.norm()); // 圆右下
        worldpoint.push_back(CC + 0.15 * NN.cross(G) / (NN.cross(G)).norm() - 0.15 * G / G.norm()); // 圆左下
        worldpoint.push_back(CC + 0.15 * NN.cross(G) / (NN.cross(G)).norm() + 0.15 * G / G.norm()); // 圆左上

        // worldpoint.push_back(CC + 0.15 * NN);                                 // 圆前
        for (auto wp : worldpoint)
        {
            Vector3d gg = estimator.ric[0].transpose() * (estimator.Rs[index].transpose() * (wp - estimator.Ps[index]) - estimator.tic[0]);
            cv::Point3d g;
            if (gg.z() < 0)
            {
                draw.clear();
                return false;
            }
            gg = intrinsic_ * gg;
            g.x = gg.x() / gg.z();
            g.y = gg.y() / gg.z();
            g.z = gg.z() / gg.z();
            draw.push_back(cv::Point2d(g.x, g.y));
        }
    }
    vector<cv::Point2d> imgpts;
    double judge = 0;
    double judge_ = 0;
    ps_0.push_back({0, 0, 0.16}); // 圆前
    cv::projectPoints(ps_0, rvec, tvec, intrinsic, distortion, imgpts);
    cv::Rodrigues(rvec, rot);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RR_(i, j) = rot.at<double>(i, j);
        }
    }
    for (int i = 0; i < pc.size(); i++)
    {
        judge += sqrt(((pc[i] - imgpts[i]).x * (pc[i] - imgpts[i]).x) + ((pc[i] - imgpts[i]).y * (pc[i] - imgpts[i]).y));
    }
    std::cout << "judge:" << judge << std::endl;
    // judge_ += (pw[0] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[1] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, -0.15, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[2] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{-0.15, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[3] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0.15, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[4] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0.15, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[5] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 1.15} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // if ((imgpts[0].x - imgpts[5].x) * (pc[0].x - pc[5].x) < 0)
    // {
    //     std::cout << "=============================PnP false=============================" << std::endl;;
    //     judge += 60;
    // }
    // if (judge_ < 0.5 && judge < 20 && dis < 8) // 初始化成功
    vector<cv::Point2d> input_pts1, input_pts2;
    input_pts1.push_back(pc[0]);
    input_pts1.push_back(pc[2]);
    input_pts1.push_back(pc[4]);
    input_pts2.push_back(imgpts[0]);
    input_pts2.push_back(imgpts[2]);
    input_pts2.push_back(imgpts[4]);
    if (judge < 90 && dis < 8 && estimator.choose_right(input_pts1, input_pts2)) // 初始化成功
    {
        if (is_sign_find != 0) // 找到了
        {
            cv::Mat axis_pic;
            if (!estimator.img_buf.empty())
            {
                estimator.img_buf.front().first.clone().copyTo(axis_pic);
                cv::cvtColor(axis_pic, axis_pic, cv::COLOR_BGR2RGB);
                // axis_pic = estimator.img_buf.front().first;
                for (int po = 0; po < pc.size(); po++)
                {
                    // cv::circle(axis_pic, imgpts[po], 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);      // red points
                    cv::circle(axis_pic, draw[po], 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);    // blue points
                    cv::circle(axis_pic, pc[po], 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);      // green points
                    cv::line(axis_pic, draw[po], pc[po], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // line

                    // cv::line(axis_pic, imgpts[0], imgpts[3], cv::Scalar(0, 0, 255), 3, cv::LINE_AA); // X-axis
                    // cv::line(axis_pic, imgpts[0], imgpts[4], cv::Scalar(0, 255, 0), 3, cv::LINE_AA); // Y-axis
                    // cv::line(axis_pic, imgpts[0], imgpts[9], cv::Scalar(255, 0, 0), 3, cv::LINE_AA); // Z-axis
                }
                // // 写入文件
                // static int e_num;
                // string tmp_str;
                // tmp_str = std::to_string(e_num);
                // char filename[50];
                // sprintf(filename, "/home/scott/gvins_yolo_output/axis-ellipse/%d.jpg", e_num);
                // imwrite(filename, axis_pic);
                // e_num++;
                printf("\033[1;35m correct ellipse!!! \n\033[0m");
            }
            SIGN temp_sign;
            temp_sign.signclass = classofsign;
            temp_sign.C = np;
            NN.normalize();
            temp_sign.N = estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
            if (id != -1)
            {
                Vector3d NN_ = estimator.Rs[k] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
                double error_eN = (180 / 3.1415926) * acos(temp_sign.N.dot(NN)) / (NN.norm() * temp_sign.N.norm());
                double error_eD = (estimator.ric[0].transpose() * (estimator.Rs[index].transpose() * (np - CC) - estimator.tic[0])).z();
                printf("\033[1;35m error_eN = %f , error_eD = %f!!! \n\033[0m", error_eN, error_eD);
                ofstream sign_path_file("/media/scott/KINGSTON/20240307data/sign.txt", ios::app);
                sign_path_file.setf(ios::fixed, ios::floatfield);
                sign_path_file.precision(9);
                sign_path_file << "class:" << classofsign << ",";
                sign_path_file << "error_eN," << error_eN << ", "
                               << "error_eD," << error_eD << ","
                               << "real_depth" << (estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (CC)-estimator.tic[0])).z() << endl;
                sign_path_file.close();
            }
            // temp_sign.C = CC;
            // NN.normalize();
            // temp_sign.N = NN;
            temp_sign.q = Quaterniond(estimator.Rs[index] * estimator.ric[0] * RR_);
            temp_sign.cvPoints.clear();

            // sign_e.C = np;
            // sign_e.N = estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
            sign_e.C = CC;
            sign_e.N = NN;
            sign_e.q = Quaterniond(estimator.Rs[index] * estimator.ric[0] * RR_);
            sign_e.cvPoints.clear();
            for (size_t i = 0; i < pc.size(); i++)
            {
                temp_sign.cvPoints.push_back(Vector2d{pc[i].x, pc[i].y});
                sign_e.cvPoints.push_back(Vector2d{pc[i].x, pc[i].y});
            }
            temp_sign.ric = estimator.ric[0];
            temp_sign.tic = estimator.tic[0];
            temp_sign.dis = dis;
            temp_sign.similarty = similarty;
            temp_sign.scale = 0.15;
            estimator.mapforsign.push_back(temp_sign);
            sign_e.ric = estimator.ric[0];
            sign_e.tic = estimator.tic[0];
            sign_e.dis = dis;
            sign_e.similarty = similarty;
            sign_e.scale = 0.15;
        }
        if (id == -1) // 管理器和地图中都没有这个标志，新标志，需要加入到地图和管理器中，但只在初始化成功的情况下写
        {
            // estimator.map_manager.initialSign(id_, classofsign, np, (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})), time, uv, is_sign_find); // 初始化路标
            // estimator.para_sign_Pose[id_][0] = np.x();                                                                                                             // 中心
            // estimator.para_sign_Pose[id_][1] = np.y();
            // estimator.para_sign_Pose[id_][2] = np.z();
            // estimator.para_sign_Pose[id_][3] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).x(); // 法向量
            // estimator.para_sign_Pose[id_][4] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).y();
            // estimator.para_sign_Pose[id_][5] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).z();
            // printf("\033[1;36mNo Data Association!\n\033[0m");
            // return false;
            // {
            //     ofstream f("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv", ios::app);
            //     f.setf(ios::fixed, ios::floatfield);
            //     if (!f)
            //     {
            //         std::cout << "打开失败！请重试！" << std::endl;;
            //         return false;
            //     }
            //     else
            //     {
            //         f.precision(9);
            //         f << id_ << ","
            //           << classofsign << ","
            //           << np[0] << ","
            //           << np[1] << ","
            //           << np[2] << ","
            //           << NN.x() << ","
            //           << NN.y() << ","
            //           << NN.z()
            //           << ","
            //           << "0.15"
            //           << ","
            //           << time << std::endl;;
            //         f.close();
            //         printf("\033[1;35m写入圆形标志数据!\n\033[0m");
            //     }
            // }
        }
        std::vector<cv::Point3d>().swap(ps_0);
        std::vector<Vector3d>().swap(ps);
        std::vector<Vector3d>().swap(pw);
        std::vector<cv::Point2d>().swap(pc);
        std::vector<cv::Point2d>().swap(draw);
        return true;
    }
    else
    {
        std::vector<cv::Point3d>().swap(ps_0);
        std::vector<Vector3d>().swap(ps);
        std::vector<Vector3d>().swap(pw);
        std::vector<cv::Point2d>().swap(pc);
        std::vector<cv::Point2d>().swap(draw);
        return false;
    }
}

// 输入：椭圆观测
// 输出：坐标，时间戳，单应矩阵
//  20221104xjl
bool processEllipse(queue<sensor_msgs::PointCloudConstPtr> &ellipse_buf, Vector3d &ep, Vector3d &np, double &time, Eigen::Matrix3d &RR_, Eigen::Vector3d &PP_, vector<pair<pair<Eigen::Vector3d, Eigen::Vector3d>, double>> &p_list)
{
    double u, v;
    double a, b;
    double ori;
    double ax, ay, bx, by;
    double m, n;
    double similarty;
    // double time;
    // int id;
    double dis;                   // 相机光心与标志中心距离
    double true_ellipse_d = 0.15; // 真实半径15cm
    string classofsign;
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 1765.682901, 0.000000, 782.352086, 0.000000, 1758.799034, 565.999397, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.060942, 0.058542, 0.001478, 0.002002);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
    //                     cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);
    Matrix3d intrinsic_;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    Vector3d p;
    double fx, fy, cx, cy, k1, k2, p1, p2;
    double in;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);

    if (!ellipse_buf.empty())
    {
        time = ellipse_buf.front()->header.stamp.toSec();

        // printf("***%f",time);
        // std::cout << estimator.Headers[WINDOW_SIZE] << std::endl;;

        classofsign = ellipse_buf.front()->header.frame_id;
        // u = ellipse_buf.front()->channels();
        p.x() = ellipse_buf.front()->points[0].x; // 去畸变后的坐标
        p.y() = ellipse_buf.front()->points[0].y;
        p.z() = ellipse_buf.front()->points[0].z; // z=1，归一化

        u = ellipse_buf.front()->channels[0].values[0]; // u,v
        v = ellipse_buf.front()->channels[1].values[0];
        a = ellipse_buf.front()->channels[2].values[0]; // a,b
        b = ellipse_buf.front()->channels[3].values[0];
        ori = ellipse_buf.front()->channels[4].values[0];       // ori=水平到长轴的夹角，x轴起始，逆时针为正
        similarty = ellipse_buf.front()->channels[5].values[0]; // 评分

        ax = u + a * cos(ori * M_PI / 180.0);
        ay = v + a * sin(ori * M_PI / 180.0);
        bx = u - a * cos(ori * M_PI / 180.0);
        by = v - a * sin(ori * M_PI / 180.0);
        // std::cout << "u:" << u << std::endl;;
        // std::cout << "v:" << v << std::endl;;
        // std::cout << "ax:" << ax << std::endl;;
        // std::cout << "ay:" << ay << std::endl;;
        // std::cout << "bx:" << bx << std::endl;;
        // std::cout << "by:" << by << std::endl;;
        ellipse_buf.pop();
    }
    else
    {
        return false;
    }
    // 20230702_xjl
    // 取出椭圆队列中的对应信息，处理步骤如下：
    // 1.根据半长轴推断深度，并根据深度和焦距的比值，求出标志的相机坐标系坐标
    // 2.根据重力投影得到图像中的横轴纵轴关键点
    // 3.根据载体位置取得关键点的先验信息，并投影到归一化平面求得投影与观测的残差
    // 4.输入对应的点对求单应矩阵

    // 求深度
    in = sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy) + (fx + fy) * (fx + fy) / 4);
    m = sqrt((fx + fy) * (fx + fy) / 4 + (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy));
    n = sqrt((fx + fy) * (fx + fy) / 4 + (bx - cx) * (bx - cx) + (by - cy) * (by - cy));
    dis = true_ellipse_d / tan(0.5 * acos((m * m + n * n - 4 * a * a) / (2 * m * n)));
    dis = 1 / dis * 0.2 * true_ellipse_d + dis;
    double nn = dis / in;

    ep.x() = (u - cx) * nn;
    ep.y() = (v - cy) * nn;
    ep.z() = (fx + fy) / 2 * nn;
    // if (abs(ep.norm() - dis)<0.1)
    //     std::cout << "result equals" << std::endl;;
    // Vector3d ep1,ep2;
    // ep1.x() = ax*nn;
    // ep1.x() = ax*nn;
    // ep1.z() = ((fx + fy) / 2) * nn*nn;
    int index;                                      // 标志对应的滑窗位置索引
    if (!estimator.sign2local(time, index, ep, np)) // 得到标志中心在世界坐标系下坐标
        return false;
    // return true;
    printf("\033[1;31m ellipse:%f\nc_pos_ellipse: %f, %f, %f \nn_pos_ellipse:%f,%f,%f\n\033[0m",
           dis, ep.x(), ep.y(), ep.z(), np.x(), np.y(), np.z());

    // csv读文件
    // 找到重力对应的投影
    Vector3d g;
    vector<Vector2d> uv;
    Vector3d NN;
    Vector3d CC = np;
    std::vector<Vector3d> ps, pw; // 相机系下的坐标,世界系下的坐标
    std::vector<cv::Point2d> pc;  // 像素坐标（深度为焦距）
    // pair<pair<Vector3d, double>, double> pp;
    if (!find_gravity(index, g)) // 找到图像对应时间戳下重力g转为相机系下的g
    {
        return false;
    }
    g.normalize();
    // 输入：在相机系下的重力g
    // 标志中心像素坐标p
    // 标志中心点相机系下位置ep
    // 长短轴、倾斜角以及
    // 得到相机平面的标志最高点、最低点、最远点、最近点(其中以靠左边的这个点，对应标志靠右的点为x轴)
    estimator.computeEllipseLineIntersection(g, cv::Point2d(u, v), ep, a, b, ori, pc, ps, pw, intrinsic, index, dis, NN); // g投影到相机平面建立坐标系，与椭圆相交并且得出xy轴的交点(这是一个数学过程)
    int id = -1;
    int id_ = 0;
    int is_sign_find = 0;
    estimator.map_manager.addSignCheck(np, classofsign, id); // 在管理器中寻找是否有同一个标志

    for (int i = 0; i < pc.size(); i++)
    {
        uv.push_back({(pc[i].x - cx) / fx, (pc[i].y - cy) / fy});
        // std::cout<<(pc[i].x - cx) / fx<<","<< (pc[i].y - cy) / fy<<std::endl;;
    }
    if (id != -1) // 若在管理器中找到了该标志
    {
        printf("find it in map_manager!\n");
        is_sign_find = 1;
        if (!estimator.map_manager.sign.empty())
        {
            for (auto it = estimator.map_manager.sign.begin(), it_next = estimator.map_manager.sign.begin(); it != estimator.map_manager.sign.end(); it = it_next) // 遍历滑窗标志
            {
                if (it->classify == classofsign && ((it->C_ - np).norm() < 2) && id == it->sign_id)
                {
                    estimator.map_manager.initialSign(id, classofsign, it->C_, it->N_, time, uv, is_sign_find);
                    NN = it->N_;
                    NN.normalize();
                    CC = it->C_;
                    break;
                    // std::cout<<it->C_<<std::endl;;
                    // std::cout<<it->N_<<std::endl;;
                }
                it_next++;
            }
        }
    }
    else // 管理器中没有，去csv文件中找
    {
        if (parseVectorMap(id, np, classofsign, p_list, time, id_)) // 使用id读地图得到所需要的标志的先验信息p_list
        {
            // 此时的id不为-1
            if (!p_list.empty())
            {
                is_sign_find = 2;
                // Vector3d angles = p_list.front().first.second;
                NN = p_list.front().first.second;
                NN.normalize();
                CC = p_list.front().first.first;
                estimator.map_manager.initialSign(id, classofsign, p_list.front().first.first, NN, time, uv, is_sign_find); // 初始化路标到管理器中，只负责加入，和add features一样
            }
        }
    }
    std::vector<cv::Point3d> ps_0;
    ps_0.push_back({0, 0, 0.01});     // 路标坐标系下的关键点
    ps_0.push_back({0, -0.15, 0.01}); // 圆上
    ps_0.push_back({-0.15, 0, 0.01}); // 圆左
    ps_0.push_back({0, 0.15, 0.01});  // 圆下
    ps_0.push_back({0.15, 0, 0.01});  // 圆右
    // ps_0.push_back({0, 0, 0.16});  // 圆前
    cv::Mat rvec(3, 1, CV_64FC1);
    cv::Mat tvec(3, 1, CV_64FC1);
    cv::Mat rot;
    // cv::solvePnP(ps_0, pc, intrinsic, distortion, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    cv::solvePnPRansac(ps_0, pc,
                       intrinsic, distortion,
                       rvec, tvec, false, 100, 5.0, 0.99);
    /******************************************************************/
    // vector<cv::Mat> possibleRvecs, possibleTvecs;
    // cv::Mat inliers;
    // cv::Mat bestRvec, bestTvec;
    // vector<cv::Mat> rvecCandidates, tvecCandidates;
    // // Use RANSAC to find possible solutions multiple times
    // int iterations = 100;
    // double minReprojectionError = numeric_limits<double>::max();
    // for (int i = 0; i < iterations; ++i)
    // {
    //     // cv::Mat rvec, tvec;
    //     bool success = solvePnPRansac(ps_0, pc, intrinsic, distortion, rvec, tvec, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
    //     if (success && inliers.rows >= 4)
    //     {
    //         double reprojectionError = norm(inliers, cv::NORM_L2);
    //         rvecCandidates.push_back(rvec);
    //         tvecCandidates.push_back(tvec);
    //         if (reprojectionError < minReprojectionError)
    //         {
    //             minReprojectionError = reprojectionError;
    //             bestRvec = rvec;
    //             bestTvec = tvec;
    //         }
    //     }
    // }
    // // Sort solutions by reprojection error
    // vector<pair<double, pair<cv::Mat, cv::Mat>>> candidates;
    // for (size_t i = 0; i < rvecCandidates.size(); ++i)
    // {
    //     vector<cv::Point2d> projectedPoints;
    //     cv::projectPoints(ps_0, rvecCandidates[i], tvecCandidates[i], intrinsic, distortion, projectedPoints);
    //     double error = norm(pc, projectedPoints, cv::NORM_L2);
    //     vector<cv::Point2d> input_pts1, input_pts2;
    //     input_pts1.push_back(pc[0]);
    //     input_pts1.push_back(pc[2]);
    //     input_pts1.push_back(pc[4]);
    //     input_pts2.push_back(projectedPoints[0]);
    //     input_pts2.push_back(projectedPoints[2]);
    //     input_pts2.push_back(projectedPoints[4]);
    //     if (estimator.choose_right(input_pts1, input_pts2))
    //     {
    //         candidates.push_back(make_pair(error, make_pair(rvecCandidates[i], tvecCandidates[i])));
    //     }
    // }
    // sort(candidates.begin(), candidates.end(), [](const pair<double, pair<cv::Mat, cv::Mat>> &a, const pair<double, pair<cv::Mat, cv::Mat>> &b)
    //      { return a.first < b.first; });
    // // Return the top 2 solutions
    // if (candidates.size() > 0)
    // {
    //     possibleRvecs.push_back(candidates[0].second.first);
    //     possibleTvecs.push_back(candidates[0].second.second);
    // }
    // if (candidates.size() > 1)
    // {
    //     possibleRvecs.push_back(candidates[1].second.first);
    //     possibleTvecs.push_back(candidates[1].second.second);
    // }
    // std::cout << "PnPsize:" << possibleRvecs.size() << std::endl;;
    /******************************************************************/
    /*****************draw key points of sign******************/
    vector<cv::Point2d> draw;
    if (is_sign_find != 0)
    {
        vector<Vector3d> worldpoint;
        worldpoint.push_back(CC);
        worldpoint.push_back(CC - 0.15 * G / G.norm());                       // 圆下
        worldpoint.push_back(CC - 0.15 * NN.cross(G) / (NN.cross(G)).norm()); // 圆左
        worldpoint.push_back(CC + 0.15 * G / G.norm());                       // 圆上
        worldpoint.push_back(CC + 0.15 * NN.cross(G) / (NN.cross(G)).norm()); // 圆右
        // worldpoint.push_back(CC + 0.15 * NN);                                 // 圆前
        for (auto wp : worldpoint)
        {
            Vector3d gg = estimator.ric[0].transpose() * (estimator.Rs[index].transpose() * (wp - estimator.Ps[index]) - estimator.tic[0]);
            cv::Point3d g;
            if (gg.z() < 0)
            {
                draw.clear();
                return false;
            }
            gg = intrinsic_ * gg;
            g.x = gg.x() / gg.z();
            g.y = gg.y() / gg.z();
            g.z = gg.z() / gg.z();
            draw.push_back(cv::Point2d(g.x, g.y));
        }
        if (draw[1].y < draw[3].y)
        {
            cv::Point2d temp;
            temp = draw[1];
            draw[1] = draw[3];
            draw[3] = temp;
        }
        if (draw[2].x < draw[4].x)
        {
            cv::Point2d temp;
            temp = draw[2];
            draw[2] = draw[4];
            draw[4] = temp;
        }
    }
    vector<cv::Point2d> imgpts;
    double judge = 0;
    double judge_ = 0;
    ps_0.push_back({0, 0, 0.16}); // 圆前
    cv::projectPoints(ps_0, rvec, tvec, intrinsic, distortion, imgpts);
    cv::Rodrigues(rvec, rot);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RR_(i, j) = rot.at<double>(i, j);
        }
    }
    for (int i = 0; i < pc.size(); i++)
    {
        judge += sqrt(((pc[i] - imgpts[i]).x * (pc[i] - imgpts[i]).x) + ((pc[i] - imgpts[i]).y * (pc[i] - imgpts[i]).y));
    }
    // std::cout << "judge:" << judge << std::endl;;

    judge_ += (pw[0] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    judge_ += (pw[1] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, -0.15, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    judge_ += (pw[2] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{-0.15, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    judge_ += (pw[3] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0.15, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    judge_ += (pw[4] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0.15, 0, 1} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[5] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 1.15} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // if ((imgpts[0].x - imgpts[5].x) * (pc[0].x - pc[5].x) < 0)
    // {
    //     std::cout << "=============================PnP false=============================" << std::endl;;
    //     judge += 60;
    // }
    // if (judge_ < 0.5 && judge < 20 && dis < 8) // 初始化成功
    vector<cv::Point2d> input_pts1, input_pts2;
    input_pts1.push_back(pc[0]);
    input_pts1.push_back(pc[2]);
    input_pts1.push_back(pc[4]);
    input_pts2.push_back(imgpts[0]);
    input_pts2.push_back(imgpts[2]);
    input_pts2.push_back(imgpts[4]);
    if (judge < 50 && dis < 8 && estimator.choose_right(input_pts1, input_pts2)) // 初始化成功
    {
        if (is_sign_find != 0) // 找到了
        {
            // cv::Mat axis_pic;
            // if (!estimator.img_buf.empty())
            // {
            // axis_pic = estimator.img_buf.front().first;
            // estimator.img_buf.front().first.clone().copyTo(axis_pic);
            // for (int po = 0; po < draw.size(); po++)
            // {
            //     cv::circle(axis_pic, draw[po], 3, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);        // blue points
            //     cv::circle(axis_pic, pc[po], 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);          // green points
            //     cv::line(axis_pic, draw[po], pc[po], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);     // line
            //     cv::line(axis_pic, imgpts[0], imgpts[3], cv::Scalar(0, 0, 255), 3, cv::LINE_AA); // X-axis
            //     cv::line(axis_pic, imgpts[0], imgpts[4], cv::Scalar(0, 255, 0), 3, cv::LINE_AA); // Y-axis
            //     cv::line(axis_pic, imgpts[0], imgpts[5], cv::Scalar(255, 0, 0), 3, cv::LINE_AA); // Z-axis
            // }
            // // 写入文件
            // static int e_num;
            // string tmp_str;
            // tmp_str = std::to_string(e_num);
            // char filename[50];
            // sprintf(filename, "/home/scott/gvins_yolo_output/axis/%d.jpg", e_num);
            // imwrite(filename, axis_pic);
            // e_num++;
            // printf("\033[1;35m correct ellipse!!! \n\033[0m");
            // }
            SIGN temp_sign;
            temp_sign.signclass = classofsign;
            temp_sign.C = np;
            NN.normalize();
            temp_sign.N = estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
            temp_sign.q = Quaterniond(estimator.Rs[index] * estimator.ric[0] * RR_);
            // std::cout << temp_sign.N << std::endl;;
            // std::cout << estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1}) << std::endl;;
            // RR_ << (Vector3d{0, 0, -1}).cross(NN).x(), (Vector3d{0, 0, -1}).cross(NN).y(), (Vector3d{0, 0, -1}).cross(NN).z(), 0, 0, -1, NN.x(), NN.y(), NN.z();
            // if (0.5 < (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).dot(NN) && 0.2 > abs((estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).dot(G)))
            // {
            //     std::cout << ".dot(N): " << (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).dot(NN) << std::endl;;
            //     std::cout << ".dot(G): " << abs((estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).dot(G)) << std::endl;;
            // }
            // else
            // {
            //     temp_sign.N = NN;
            //     Eigen::Vector3d k = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1}));
            //     Eigen::Vector3d k1 = k.cross(NN);
            //     double c = k.dot(NN);
            //     // 计算正弦值
            //     double s = sqrt(1 - c * c);
            //     // 构造旋转矩阵
            //     Eigen::Matrix3d Rnn;
            //     Rnn = Eigen::Matrix3d::Identity() + (1 - c) * (k * k.transpose() - Eigen::Matrix3d::Identity()) / k.norm();
            //     temp_sign.q = Quaterniond(estimator.Rs[index] * estimator.ric[0] * RR_ * Rnn.transpose());
            // }
            temp_sign.cvPoints.clear();
            for (size_t i = 0; i < pc.size(); i++)
            {
                temp_sign.cvPoints.push_back(Vector2d{pc[i].x, pc[i].y});
            }
            temp_sign.ric = estimator.ric[0];
            temp_sign.tic = estimator.tic[0];
            temp_sign.dis = dis;
            temp_sign.similarty = similarty;
            temp_sign.scale = 0.15;
            estimator.mapforsign.push_back(temp_sign);
        }
        if (id == -1) // 管理器和地图中都没有这个标志，新标志，需要加入到地图和管理器中，但只在初始化成功的情况下写
        {
            // estimator.map_manager.initialSign(id_, classofsign, np, (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})), time, uv, is_sign_find); // 初始化路标
            // estimator.para_sign_Pose[id_][0] = np.x();                                                                                                             // 中心
            // estimator.para_sign_Pose[id_][1] = np.y();
            // estimator.para_sign_Pose[id_][2] = np.z();
            // estimator.para_sign_Pose[id_][3] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).x(); // 法向量
            // estimator.para_sign_Pose[id_][4] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).y();
            // estimator.para_sign_Pose[id_][5] = (estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1})).z();
            // printf("\033[1;36mNo Data Association!\n\033[0m");
            // {
            //     ofstream f("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv", ios::app);
            //     f.setf(ios::fixed, ios::floatfield);
            //     if (!f)
            //     {
            //         std::cout << "打开失败！请重试！" << std::endl;;
            //         return false;
            //     }
            //     else
            //     {
            //         f.precision(9);
            //         f << id_ << ","
            //           << classofsign << ","
            //           << np[0] << ","
            //           << np[1] << ","
            //           << np[2] << ","
            //           << NN.x() << ","
            //           << NN.y() << ","
            //           << NN.z()
            //           << ","
            //           << "0.15"
            //           << ","
            //           << time << std::endl;;
            //         f.close();
            //         printf("\033[1;35m写入圆形标志数据!\n\033[0m");
            //     }
            // }
        }
    }
    std::vector<cv::Point3d>().swap(ps_0);
    std::vector<Vector3d>().swap(ps);
    std::vector<Vector3d>().swap(pw);
    std::vector<cv::Point2d>().swap(pc);
    std::vector<cv::Point2d>().swap(draw);
    return true;
}

bool process_Tri(SIGN &sign_t, cv::Mat intrinsic, cv::Mat distortion, int k)
{
    vector<pair<pair<Eigen::Vector3d, Eigen::Vector3d>, double>> p_list;

    SIGN sign_tri;
    double u, v; // 中心点
    double a1, b1;
    double a2, b2;
    double a3, b3; // 三顶点
    double time;
    string classoftri; // 类别
    double similarty;  // 置信度
    double check;
    std::vector<cv::Point3d> srcPoints, dstPoints;
    std::vector<cv::Point2d> pc, draw;
    static std::vector<cv::Point2d> pre_pc;
    static double last_time;
    bool activate;
    double dis;               // 距离
    double true_tri_d = 0.30; // 真实边长30cm
    Matrix3d intrinsic_;
    int is_sign_find = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    double fx, fy, cx, cy, k1, k2, p1, p2;
    double in;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);

    // if (!tri_buf.empty())
    // {
    time = sign_t.time;
    classoftri = sign_t.signclass;
    u = sign_t.features[0];
    v = sign_t.features[1];
    a1 = sign_t.features[2];
    b1 = sign_t.features[3];
    a2 = sign_t.features[4];
    b2 = sign_t.features[5];
    a3 = sign_t.features[6];
    b3 = sign_t.features[7];
    similarty = sign_t.features[10];

    pc.push_back(cv::Point2d(u, v)); // 中左下右
    pc.push_back(cv::Point2d(a1, b1));
    pc.push_back(cv::Point2d(a2, b2));
    pc.push_back(cv::Point2d(a3, b3));

    // }
    Vector3d ep, np; // 相机系和世界系下中心点，并求法向量
    std::vector<cv::Point2d> imgpts;
    double judge = 0;
    double judge_ = 0;
    dstPoints.push_back(cv::Point3d(0, 0, 0.01)); // 中左下右上
    dstPoints.push_back(cv::Point3d(0 + 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0.01));
    dstPoints.push_back(cv::Point3d(0, 0 + true_tri_d * 0.5773502, 0.01));
    dstPoints.push_back(cv::Point3d(0 - 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0.01));
    dstPoints.push_back(cv::Point3d(0, 0 - true_tri_d * 0.28867513, 0.01));
    cv::Mat rvec(3, 1, CV_64FC1);
    cv::Mat tvec(3, 1, CV_64FC1); // 相机坐标到路标坐标系的坐标系变换
    // csv读文件
    Vector3d g;
    // 找到重力对应的投影
    if (!find_gravity(k, g)) // 找到图像对应时间戳下重力g转为相机系下的g
    {
        return false;
    }
    g.normalize();
    vector<Vector2d> uv; // 归一化
    static vector<Vector2d> last_uv;
    std::vector<Vector3d> ps, pw; // 相机系下的坐标,世界系下的坐标
    Vector3d n, pe;               // 两帧计算得出的相机系下的法向量，位置
    Vector3d NN;
    Vector3d CC;
    dis = estimator.computeTri(pc, ep, np, intrinsic_, true_tri_d, k, g); // 估计距离和世界系下的坐标（有误差）
    // printf("\033[1;34m c_pos_triangle:%f,%f,%f\n\033[0m", ep.x(), ep.y(), ep.z());
    // printf("\033[1;35m n_pos_triangle:%f,%f,%f\n\033[0m", np.x(), np.y(), np.z());
    if (cv::solvePnP(dstPoints, pc, intrinsic, distortion, rvec, tvec, false, cv::SOLVEPNP_EPNP)) // 解P3P
    {
        cv::projectPoints(dstPoints, rvec, tvec, intrinsic, distortion, imgpts);
        for (int i = 0; i < pc.size(); i++)
        {
            judge += abs(imgpts[i].x - pc[i].x) + abs(imgpts[i].y - pc[i].y); // 像素投影误差
        }
        if (judge > 75)
        {
            std::cout << "三角形投影累计误差超出阈值：" << judge << std::endl;
            return false;
        }
    }
    else
        return false;
    int id = -1;
    int id_ = 0;
    estimator.map_manager.addSignCheck(np, classoftri, id); // 在管理器中寻找是否有同一个标志
    CC = np;
    for (int i = 0; i < pc.size(); i++)
    {
        uv.push_back({(pc[i].x - cx) / fx, (pc[i].y - cy) / fy});
    }
    if (id != -1) // 若在管理器中找到了该标志
    {
        printf("find it in map_manager!\n");
        is_sign_find = 1;
        if (!estimator.map_manager.sign.empty())
        {
            for (auto it = estimator.map_manager.sign.begin(), it_next = estimator.map_manager.sign.begin(); it != estimator.map_manager.sign.end(); it = it_next) // 遍历滑窗标志
            {
                if (it->classify == classoftri && ((it->C_ - np).norm() < 2) && id == it->sign_id)
                {
                    if (abs(time - last_time) < 0.05) // if continous
                    {
                        // estimator.computeRT(it->sign_per_frame.back().pts, uv, pe, n);
                        Matrix4d Trw, Tcw;
                        Trw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE - 1] * estimator.ric[0];
                        Trw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE - 1] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE - 1];
                        Trw(3, 3) = 1;
                        Trw(3, 2) = 0;
                        Trw(3, 1) = 0;
                        Trw(3, 0) = 0;
                        // std::cout << Trw << std::endl;;
                        Tcw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE] * estimator.ric[0];
                        Tcw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE];
                        Tcw(3, 3) = 1;
                        Tcw(3, 2) = 0;
                        Tcw(3, 1) = 0;
                        Tcw(3, 0) = 0;
                        // std::cout << Tcw << std::endl;;
                        int idx;
                        if (estimator.GetSignTheta(last_uv, uv, pre_pc, pc, Trw, Tcw, idx, n, intrinsic_))
                        {
                            n.normalize();
                            // std::cout << "n:" << n << std::endl;;
                        }
                    }
                    estimator.map_manager.initialSign(id, classoftri, it->C_, it->N_, time, uv, is_sign_find); // 在管理器中初始化一个标志
                    NN = it->N_;
                    NN.normalize();
                    CC = it->C_;
                    break;
                }
                it_next++;
            }
        }
    }
    else // 管理器中没有，去csv文件中找
    {
        if (parseVectorMap(id, np, classoftri, p_list, time, id_)) // 使用id读地图得到所需要的标志的先验信息p_list
        {
            // 此时的id不为-1
            if (!p_list.empty() && id != 1)
            {
                is_sign_find = 2;
                NN = p_list.front().first.second;
                CC = p_list.front().first.first;
                estimator.map_manager.initialSign(id, classoftri, p_list.front().first.first, NN, time, uv, is_sign_find); // 初始化路标到管理器中，只负责加入，和add features一样
                p_list.clear();
            }
        }
    }
    Matrix3d RR_;
    Vector3d t_;
    cv::Mat rot;
    cv::Rodrigues(rvec, rot); // 相机到路标系
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RR_(i, j) = rot.at<double>(i, j);
        }
    }
    t_.x() = tvec.at<double>(0, 0);
    t_.y() = tvec.at<double>(1, 0);
    t_.z() = tvec.at<double>(2, 0);
    // std::cout << "*********************" << t_.x() * (RR_ * Vector3d{0, 0, 1}).x() + t_.y() * (RR_ * Vector3d{0, 0, 1}).y() + t_.z() * (RR_ * Vector3d{0, 0, 1}).z() << std::endl;;
    // std::cout << "t:" << t_ << std::endl;;
    if (id == -1)
    {
        NN = estimator.Rs[k] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
        CC = estimator.Rs[k] * (estimator.ric[0] * t_ + estimator.tic[0]) + estimator.Ps[k];
    }
    printf("\033[1;36m dis of triangle:%f\nc_pos_triangle: %f, %f, %f \nn_pos_triangle:%f,%f,%f\nn_thita_triangle:%f,%f,%f\n\033[0m",
           dis, t_.x(), t_.y(), t_.z(), CC.x(), CC.y(), CC.z(), NN.x(), NN.y(), NN.z());
    if (id != -1)
    {
        Vector3d NN_ = estimator.Rs[k] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
        double error_tN = (180 / 3.1415926) * acos(NN_.dot(NN)) / (NN.norm() * NN_.norm());
        double error_tD = (estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (np - CC) - estimator.tic[0])).z();
        printf("\033[1;35m error_tN = %f , error_tD = %f!!! \n\033[0m", error_tN, error_tD);
        ofstream sign_path_file("/media/scott/KINGSTON/20240307data/sign.txt", ios::app);
        sign_path_file.setf(ios::fixed, ios::floatfield);
        sign_path_file.precision(9);
        sign_path_file << "class:" << classoftri << ",";
        sign_path_file << "error_tN," << error_tN << ", "
                       << "error_tD," << error_tD << ","
                       << "real_depth" << (estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (CC)-estimator.tic[0])).z() << endl;
        sign_path_file.close();
    }

    vector<Vector3d> worldpoint;
    // // 世界系下的标志投影回到相机系
    worldpoint.push_back(CC);                                                                                                  // 中
    worldpoint.push_back(CC + 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.28867513 * G / G.norm()); // 左
    worldpoint.push_back(CC - true_tri_d * 0.5773502 * G / G.norm());                                                          // 下
    worldpoint.push_back(CC - 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.28867513 * G / G.norm()); // 右
    worldpoint.push_back(CC + true_tri_d * 0.28867513 * G / G.norm());                                                         // 上
    for (auto wp : worldpoint)
    {
        int i = 0;
        Vector3d gg = estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (wp - estimator.Ps[k]) - estimator.tic[0]);
        cv::Point3d g;
        if (gg.z() < 0)
            return false;
        // gg = intrinsic_ * gg;
        g.x = gg.x() / gg.z();
        g.y = gg.y() / gg.z();
        g.z = gg.z() / gg.z();
        draw.push_back(cv::Point2d(g.x, g.y));
        judge_ += abs(uv[i].x() - g.x) + abs(uv[i].y() - g.y);
        i++;
    }
    // std::cout << "judge_" << judge_ << std::endl;
    // 中左下右
    // if (judge < 10 && judge_ < 0.5 && dis < 8 && tvec.at<double>(0, 2) > 0)
    if (judge < 90 && dis < 8 && tvec.at<double>(0, 2) > 0)
    {
        vector<cv::Point2d> input_pts1, input_pts2, input_pts3;
        input_pts1.push_back(pc[0]);
        input_pts1.push_back(pc[1]);
        input_pts1.push_back(pc[3]);
        input_pts2.push_back(draw[0]);
        input_pts2.push_back(draw[1]);
        input_pts2.push_back(draw[3]);
        input_pts3.push_back(imgpts[0]);
        input_pts3.push_back(imgpts[1]);
        input_pts3.push_back(imgpts[3]);
        if (!estimator.choose_right(input_pts1, input_pts2) || !estimator.choose_right(input_pts1, input_pts3))
        {
            std::cout << "wrong triangle answer!!!" << std::endl;
            return false;
        }
        else
        {
            std::cout << "right triangle answer!!!" << std::endl;
        }
        // 初始化标志到标志管理器和csv
        Matrix3d Rsw;
        Rsw = (estimator.Rs[k] * estimator.ric[0] * RR_).transpose();
        // std::cout << (estimator.Rs[index] * estimator.ric[0] * RR_).transpose() << std::endl;;
        SIGN temp_sign;
        temp_sign.signclass = classoftri;
        temp_sign.C = estimator.Rs[k] * (estimator.ric[0] * t_ + estimator.tic[0]) + estimator.Ps[k];
        NN.z() = 0;
        NN.normalize();
        temp_sign.N = NN;
        temp_sign.q = Quaterniond(estimator.Rs[k] * estimator.ric[0] * RR_);
        temp_sign.cvPoints.clear();
        for (int i = 0; i < pc.size(); i++)
        {
            temp_sign.cvPoints.push_back(Vector2d{uv[i].x(), uv[i].y()});
            sign_t.cvPoints.push_back(Vector2d{pc[i].x, pc[i].y});
        }
        temp_sign.ric = estimator.ric[0];
        temp_sign.tic = estimator.tic[0];
        temp_sign.dis = dis;
        temp_sign.similarty = similarty;
        temp_sign.scale = 0.30;

        sign_t.signclass = classoftri;
        sign_t.C = CC;
        sign_t.N = temp_sign.N;
        sign_t.q = Quaterniond(estimator.Rs[k] * estimator.ric[0] * RR_);
        sign_t.ric = estimator.ric[0];
        sign_t.tic = estimator.tic[0];
        sign_t.dis = dis;
        sign_t.similarty = similarty;
        sign_t.scale = 0.30;
        estimator.mapforsign.push_back(temp_sign);
        std::cout << "add triagnle!!!" << std::endl;

        if (id == -1)
        {
            // 计算在世界系下的法向量和中心点
            // estimator.map_manager.initialSign(id_, classoftri, CC, NN, time, uv, is_sign_find); // 初始化路标
            // estimator.para_sign_Pose[id_][0] = CC.x();                                          // 中心
            // estimator.para_sign_Pose[id_][1] = CC.y();
            // estimator.para_sign_Pose[id_][2] = CC.z();
            // estimator.para_sign_Pose[id_][3] = NN.x(); // 法向量
            // estimator.para_sign_Pose[id_][4] = NN.y();
            // estimator.para_sign_Pose[id_][5] = NN.z();

            //     printf("\033[1;32m No Data Association!\n\033[0m");
            //     {
            //         ofstream f("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/", ios::app);
            //         f.setf(ios::fixed, ios::floatfield);
            //         if (!f)
            //         {
            //             std::cout << "打开失败！请重试！" << std::endl;;
            //             return false;
            //         }
            //         else
            //         {
            //             f.precision(9);
            //             f << id_ << ","
            //               << classoftri << ","
            //               << CC[0] << ","
            //               << CC[1] << ","
            //               << CC[2] << ","
            //               << NN.x() << ","
            //               << NN.y() << ","
            //               << NN.z()
            //               << ","
            //               << "0.30"
            //               << ","
            //               << time << std::endl;;
            //             f.close();
            //             printf("\033[1;35m 写入三角形标志数据!\n\033[0m");
            //         }
            //     }
        }
        cv::Mat axis_pic;
        if (!estimator.img_buf.empty())
        {
            // axis_pic = estimator.img_buf.front().first;
            estimator.img_buf.front().first.clone().copyTo(axis_pic);
            cv::cvtColor(axis_pic, axis_pic, cv::COLOR_BGR2RGB);
            for (int i = 0; i < draw.size(); i++)
            {
                draw[i].x = draw[i].x * fx + cx;
                draw[i].y = draw[i].y * fy + cy;
                cv::circle(axis_pic, draw[i], 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);   // 投影
                cv::circle(axis_pic, pc[i], 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);     // 检测点
                cv::line(axis_pic, draw[i], pc[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 点点连线
            }
            // cv::line(axis_pic, pc[0], pc[1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 点点连线
            // cv::line(axis_pic, pc[0], pc[4], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 点点连线
            // cv::line(axis_pic, pc[0], pc[3], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 点点连线

            // 写入文件
            // static int t_num;
            // string tmp_str;
            // tmp_str = std::to_string(t_num);
            // char filename[50];
            // sprintf(filename, "/home/scott/gvins_yolo_output/axis-tri/%d.jpg", t_num);
            // cv::imwrite(filename, axis_pic);
            // t_num++;
            printf("\033[1;35m correct triangle! \n\033[0m");
        }
    }
    else
    {
        return false;
    }

    last_time = time;
    pre_pc = pc;
    last_uv = uv;
    draw.clear();
    return true;
}

bool process_Rect(SIGN &sign_r, cv::Mat intrinsic, cv::Mat distortion, int k)
{
    vector<pair<pair<Eigen::Vector3d, Eigen::Vector3d>, double>> p_list;

    // SIGN sign_rect;
    double u, v; // 中心点
    double a1, b1;
    double a2, b2;
    double a3, b3;
    double a4, b4; // 四顶点
    double time;
    string classofrect; // 类别
    double similarty;   // 置信度
    double check;
    std::vector<cv::Point3d> dstPoints;
    std::vector<cv::Point2d> pc, draw;
    static std::vector<cv::Point2d> pre_pc;
    static double last_time;
    bool activate;
    double dis;               // 距离
    double true_tri_d = 0.30; // 真实边长30cm
    Matrix3d intrinsic_;
    int is_sign_find = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    double fx, fy, cx, cy, k1, k2, p1, p2;
    double in;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);

    time = sign_r.time;
    classofrect = sign_r.signclass;
    u = sign_r.features[0];
    v = sign_r.features[1];
    a1 = sign_r.features[2];
    b1 = sign_r.features[3];
    a2 = sign_r.features[4];
    b2 = sign_r.features[5];
    a3 = sign_r.features[6];
    b3 = sign_r.features[7];
    a4 = sign_r.features[8];
    b4 = sign_r.features[9];
    similarty = sign_r.features[10];

    pc.push_back(cv::Point2d(u, v));
    pc.push_back(cv::Point2d(a1, b1));                       // 右上
    pc.push_back(cv::Point2d(a2, b2));                       // 右下
    pc.push_back(cv::Point2d(a3, b3));                       // 左下
    pc.push_back(cv::Point2d(a4, b4));                       // 左上
    pc.push_back(cv::Point2d((a1 + a4) / 2, (b1 + b4) / 2)); // 上
    pc.push_back(cv::Point2d((a3 + a4) / 2, (b3 + b4) / 2)); // 左
    pc.push_back(cv::Point2d((a2 + a3) / 2, (b2 + b3) / 2)); // 下
    pc.push_back(cv::Point2d((a1 + a2) / 2, (b1 + b2) / 2)); // 右
    Vector3d ep, np;                                         // 相机系和世界系下中心点，并求法向量
    std::vector<cv::Point2d> imgpts;
    double judge = 0;
    double judge_ = 0;
    dstPoints.push_back(cv::Point3d(0, 0, 0.01));
    dstPoints.push_back(cv::Point3d(0 - 0.5 * true_tri_d, 0 - true_tri_d * 0.5, 0.01)); // 右上
    dstPoints.push_back(cv::Point3d(0 - 0.5 * true_tri_d, 0 + true_tri_d * 0.5, 0.01)); // 右下
    dstPoints.push_back(cv::Point3d(0 + 0.5 * true_tri_d, 0 + true_tri_d * 0.5, 0.01)); // 左下
    dstPoints.push_back(cv::Point3d(0 + 0.5 * true_tri_d, 0 - true_tri_d * 0.5, 0.01)); // 左上
    dstPoints.push_back(cv::Point3d(0, 0 - true_tri_d * 0.5, 0.01));                    // 上
    dstPoints.push_back(cv::Point3d(0 + 0.5 * true_tri_d, 0, 0.01));                    // 左
    dstPoints.push_back(cv::Point3d(0, 0 + true_tri_d * 0.5, 0.01));                    // 下
    dstPoints.push_back(cv::Point3d(0 - 0.5 * true_tri_d, 0, 0.01));                    // 右
    cv::Mat rvec(3, 1, CV_64FC1);
    cv::Mat tvec(3, 1, CV_64FC1); // 相机坐标到路标坐标系的坐标系变换
    // csv读文件
    Vector3d g;
    // 找到重力对应的投影
    if (!find_gravity(k, g)) // 找到图像对应时间戳下重力g转为相机系下的g
    {
        return false;
    }
    g.normalize();
    vector<Vector2d> uv; // 归一化
    static vector<Vector2d> last_uv;
    std::vector<Vector3d> ps, pw; // 相机系下的坐标,世界系下的坐标
    Vector3d n, pe;               // 两帧计算得出的相机系下的法向量，位置
    Vector3d NN;
    Vector3d CC;
    // if (cv::solvePnP(dstPoints, pc, intrinsic, distortion, rvec, tvec, false, cv::SOLVEPNP_EPNP))         // 解P3P
    if (cv::solvePnPRansac(dstPoints, pc,
                           intrinsic, distortion,
                           rvec, tvec, false, 100, 5.0, 0.99))
    {
        cv::projectPoints(dstPoints, rvec, tvec, intrinsic, distortion, imgpts);
        for (int i = 0; i < pc.size(); i++)
        {
            judge += abs(imgpts[i].x - pc[i].x) + abs(imgpts[i].y - pc[i].y); // 像素投影误差
        }
        if (judge > 90)
        {
            std::cout << "矩形投影累计误差超出阈值：" << judge << std::endl;
            return false;
        }
    }
    else
        return false;

    int id = -1;
    int id_ = 0;
    np = estimator.Rs[k] * (estimator.ric[0] * Vector3d{tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)} + estimator.tic[0]) + estimator.Ps[k];
    estimator.map_manager.addSignCheck(np, classofrect, id); // 在管理器中寻找是否有同一个标志
    CC = np;
    for (int i = 0; i < pc.size(); i++)
    {
        uv.push_back({(pc[i].x - cx) / fx, (pc[i].y - cy) / fy});
    }
    if (id != -1) // 若在管理器中找到了该标志
    {
        printf("find it in map_manager!\n");
        is_sign_find = 1;
        if (!estimator.map_manager.sign.empty())
        {
            for (auto it = estimator.map_manager.sign.begin(), it_next = estimator.map_manager.sign.begin(); it != estimator.map_manager.sign.end(); it = it_next) // 遍历滑窗标志
            {
                if (it->classify == classofrect && ((it->C_ - np).norm() < 2) && id == it->sign_id)
                {
                    if (abs(time - last_time) < 0.05) // if continous
                    {
                        // estimator.computeRT(it->sign_per_frame.back().pts, uv, pe, n);
                        Matrix4d Trw, Tcw;
                        Trw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE - 1] * estimator.ric[0];
                        Trw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE - 1] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE - 1];
                        Trw(3, 3) = 1;
                        Trw(3, 2) = 0;
                        Trw(3, 1) = 0;
                        Trw(3, 0) = 0;
                        // std::cout << Trw << std::endl;;
                        Tcw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE] * estimator.ric[0];
                        Tcw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE];
                        Tcw(3, 3) = 1;
                        Tcw(3, 2) = 0;
                        Tcw(3, 1) = 0;
                        Tcw(3, 0) = 0;
                        // std::cout << Tcw << std::endl;;
                        int idx;
                        if (estimator.GetSignTheta(last_uv, uv, pre_pc, pc, Trw, Tcw, idx, n, intrinsic_))
                        {
                            n.normalize();
                            // std::cout << "n:" << n << std::endl;;
                        }
                    }
                    estimator.map_manager.initialSign(id, classofrect, it->C_, it->N_, time, uv, is_sign_find); // 在管理器中初始化一个标志
                    NN = it->N_;
                    NN.normalize();
                    CC = it->C_;
                    break;
                }
                it_next++;
            }
        }
    }
    else // 管理器中没有，去csv文件中找
    {
        if (parseVectorMap(id, np, classofrect, p_list, time, id_)) // 使用id读地图得到所需要的标志的先验信息p_list
        {
            // 此时的id不为-1
            if (!p_list.empty() && id != 1)
            {
                is_sign_find = 2;
                NN = p_list.front().first.second;
                CC = p_list.front().first.first;
                estimator.map_manager.initialSign(id, classofrect, p_list.front().first.first, NN, time, uv, is_sign_find); // 初始化路标到管理器中，只负责加入，和add features一样
                p_list.clear();
            }
        }
    }
    Matrix3d RR_;
    Vector3d t_;
    cv::Mat rot;
    cv::Rodrigues(rvec, rot); // 相机到路标系
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RR_(i, j) = rot.at<double>(i, j);
        }
    }
    t_.x() = tvec.at<double>(0, 0);
    t_.y() = tvec.at<double>(1, 0);
    t_.z() = tvec.at<double>(2, 0);
    if (id == -1)
    {
        NN = estimator.Rs[k] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
        CC = estimator.Rs[k] * (estimator.ric[0] * t_ + estimator.tic[0]) + estimator.Ps[k];
        NN.z() = 0;
        NN.normalize();
    }
    if (id != -1)
    {
        Vector3d NN_ = estimator.Rs[k] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
        double error_rN = (180 / 3.1415926) * acos(NN_.dot(NN)) / (NN.norm() * NN_.norm());
        double error_rD = (estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (np - CC) - estimator.tic[0])).z();
        // printf("\033[1;35m error_rN = %f , error_rD = %f!!! \n\033[0m", error_rN, error_rD);
        ofstream sign_path_file("/media/scott/KINGSTON/20240307data/sign.txt", ios::app);
        sign_path_file.setf(ios::fixed, ios::floatfield);
        sign_path_file.precision(9);
        sign_path_file << "class:" << classofrect << ",";
        sign_path_file << "error_rN," << error_rN << ", " << "error_rD," << error_rD
                       << "," << "real_depth" << (estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (CC)-estimator.tic[0])).z() << endl;
        sign_path_file.close();
    }
    printf("\033[1;34m c_pos_rectangle: %f, %f, %f \nn_pos_rectangle:%f,%f,%f\nn_thita_rectangle:%f,%f,%f\n\033[0m",
           t_.x(), t_.y(), t_.z(), CC.x(), CC.y(), CC.z(), NN.x(), NN.y(), NN.z());
    if (id != -1)
    {
        vector<Vector3d> worldpoint;
        // // // 世界系下的标志投影回到相机系
        worldpoint.push_back(CC);                                                                                           // 中
        worldpoint.push_back(CC - 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.5 * G / G.norm()); // 右上
        worldpoint.push_back(CC - 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() - true_tri_d * 0.5 * G / G.norm()); // 右下
        worldpoint.push_back(CC + 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() - true_tri_d * 0.5 * G / G.norm()); // 左下
        worldpoint.push_back(CC + 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.5 * G / G.norm()); // 左上

        worldpoint.push_back(CC + true_tri_d * 0.5 * G / G.norm());                       // 上
        worldpoint.push_back(CC + 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm()); // 左
        worldpoint.push_back(CC - true_tri_d * 0.5 * G / G.norm());                       // 下
        worldpoint.push_back(CC - 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm()); // 右

        int i = 0;
        for (auto wp : worldpoint)
        {
            Vector3d gg = estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (wp - estimator.Ps[k]) - estimator.tic[0]);
            cv::Point3d g;
            if (gg.z() < 0)
                return false;
            // gg = intrinsic_ * gg;
            g.x = gg.x() / gg.z();
            g.y = gg.y() / gg.z();
            g.z = gg.z() / gg.z();
            draw.push_back(cv::Point2d(g.x, g.y));
            judge_ += abs(uv[i].x() - g.x) + abs(uv[i].y() - g.y);
            i++;
        }
    }
    // std::cout << "judge_" << judge_ << std::endl;;
    if (judge < 90 && dis < 8 && tvec.at<double>(0, 2) > 0)
    {
        // vector<cv::Point2d> input_pts1, input_pts2, input_pts3;
        // input_pts1.push_back(pc[0]); // 中心点
        // input_pts1.push_back(pc[1]); // 左上
        // input_pts1.push_back(pc[4]);
        // input_pts2.push_back(draw[0]);
        // input_pts2.push_back(draw[1]);
        // input_pts2.push_back(draw[4]);
        // input_pts3.push_back(imgpts[0]);
        // input_pts3.push_back(imgpts[1]);
        // input_pts3.push_back(imgpts[4]);
        // if (!estimator.choose_right(input_pts1, input_pts2) || !estimator.choose_right(input_pts1, input_pts3))
        // {
        //     std::cout << "wrong rectangle answer!!!" << std::endl;
        //     return false;
        // }
        // else
        // {
        //     std::cout << "right rectangle answer!!!" << std::endl;
        // }

        // Matrix3d Rsw;
        // Rsw = (estimator.Rs[k] * estimator.ric[0] * RR_).transpose();
        // std::cout << (estimator.Rs[index] * estimator.ric[0] * RR_).transpose() << std::endl;
        SIGN temp_sign;
        temp_sign.signclass = classofrect;
        temp_sign.C = estimator.Rs[k] * (estimator.ric[0] * t_ + estimator.tic[0]) + estimator.Ps[k];
        temp_sign.N = estimator.Rs[k] * estimator.ric[0] * RR_ * Vector3d{0, 0, 1};
        temp_sign.q = Quaterniond(estimator.Rs[k] * estimator.ric[0] * RR_);
        temp_sign.cvPoints.clear();
        for (int i = 0; i < pc.size(); i++)
        {
            temp_sign.cvPoints.push_back(Vector2d{uv[i].x(), uv[i].y()});
            sign_r.cvPoints.push_back(Vector2d{pc[i].x, pc[i].y});
        }
        temp_sign.ric = estimator.ric[0];
        temp_sign.tic = estimator.tic[0];
        temp_sign.dis = dis;
        temp_sign.similarty = similarty;
        temp_sign.scale = 0.3;

        sign_r.signclass = classofrect;
        sign_r.C = CC;
        sign_r.N = NN;
        sign_r.q = Quaterniond(estimator.Rs[k] * estimator.ric[0] * RR_);
        sign_r.ric = estimator.ric[0];
        sign_r.tic = estimator.tic[0];
        sign_r.dis = dis;
        sign_r.similarty = similarty;
        sign_r.scale = 0.3;
        estimator.mapforsign.push_back(temp_sign);
        std::cout << "add rectagnle!!!" << std::endl;
        if (id == -1) // 初始化标志到标志管理器和csv
        {
            // printf("\033[1;32m No Data Association!\n\033[0m");
            // // 计算在世界系下的法向量和中心点
            // estimator.map_manager.initialSign(id_, classofrect, CC, NN, time, uv, is_sign_find); // 初始化路标
            // estimator.para_sign_Pose[id_][0] = CC.x();                                           // 中心
            // estimator.para_sign_Pose[id_][1] = CC.y();
            // estimator.para_sign_Pose[id_][2] = CC.z();
            // estimator.para_sign_Pose[id_][3] = NN.x(); // 法向量
            // estimator.para_sign_Pose[id_][4] = NN.y();
            // estimator.para_sign_Pose[id_][5] = NN.z();
            // {
            //     ofstream f("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv", ios::app);
            //     f.setf(ios::fixed, ios::floatfield);
            //     if (!f)
            //     {
            //         std::cout << "打开失败！请重试！" << std::endl;;
            //         return false;
            //     }
            //     else
            //     {
            //         f.precision(9);
            //         f << id_ << ","
            //           << classofrect << ","
            //           << CC[0] << ","
            //           << CC[1] << ","
            //           << CC[2] << ","
            //           << NN.x() << ","
            //           << NN.y() << ","
            //           << NN.z()
            //           << ","
            //           << "0.30"
            //           << ","
            //           << time << std::endl;;
            //         f.close();
            //         printf("\033[1;35m 写入矩形标志数据!\n\033[0m");
            //     }
            // }
        }
        cv::Mat axis_pic;
        if (!estimator.img_buf.empty())
        {
            estimator.img_buf.front().first.clone().copyTo(axis_pic);
            cv::cvtColor(axis_pic, axis_pic, cv::COLOR_BGR2RGB);
            for (int i = 0; i < pc.size(); i++)
            {
                cv::circle(axis_pic, pc[i], 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA); // 检测点
                // cv::circle(axis_pic, pc[i], 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                // cv::circle(axis_pic, imgpts[i], 3, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                if (!draw.empty())
                {
                    // cv::circle(axis_pic, draw[i], 1, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);   // 投影
                    cv::circle(axis_pic, cv::Point2d{draw[i].x * fx + cx, draw[i].y * fy + cy}, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);   // 投影
                    cv::line(axis_pic, cv::Point2d{draw[i].x * fx + cx, draw[i].y * fy + cy}, pc[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 点点连线
                    // cv::line(axis_pic, cv::Point2d{draw[i].x * fx + cx, draw[i].y * fy + cy}, pc[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    // cout << draw[i].x << draw[i].y << endl;
                }
            }
            // 写入文件
            // static int r_num;
            // string tmp_str;
            // tmp_str = std::to_string(r_num);
            // char filename[50];
            // sprintf(filename, "/home/scott/gvins_yolo_output/axis-rect/%d.jpg", r_num);
            // cv::imwrite(filename, axis_pic);
            // r_num++;
            printf("\033[1;35m correct rectangle! \n\033[0m");
        }
    }
    else
    {
        return false;
    }

    last_time = time;
    pre_pc = pc;
    last_uv = uv;
    draw.clear();
    return true;
}

bool solveSignPnP(vector<SIGN> &signvec, cv::Mat intrinsic, cv::Mat distortion, int k)
{
    vector<cv::Point2d> imgpts, imgpts_;
    vector<cv::Point3d> realpts;
    Vector3d fivepoints;
    double fx, fy, cx, cy, k1, k2, p1, p2;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);
    for (auto sign : signvec)
    {
        if (sign.signclass[0] == 'c')
        {
            fivepoints = sign.C; // 圆中
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * G / G.norm(); // 圆下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆右
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + sign.scale * G / G.norm(); // 圆上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆左
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});

            fivepoints = sign.C + sign.scale * G / G.norm() - sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆右上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * G / G.norm() - sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆右下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * G / G.norm() + sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆左下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + sign.scale * G / G.norm() + sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 圆左上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});

            // std::cout << "sign.cvPoints.size" << sign.cvPoints.size() << std::endl;;
            for (auto cvpoint : sign.cvPoints)
            {
                imgpts.push_back(cv::Point2d{cvpoint.x(), cvpoint.y()});
            }
        }
        if (sign.signclass[0] == 't')
        {
            fivepoints = sign.C; // 中
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() + sign.scale * 0.28867513 * G / G.norm(); // 左
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * 0.5773502 * G / G.norm(); // 下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() + sign.scale * 0.28867513 * G / G.norm(); // 右
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + sign.scale * 0.28867513 * G / G.norm(); // 上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            // std::cout << "sign.cvPoints.size" << sign.cvPoints.size() << std::endl;;
            for (auto cvpoint : sign.cvPoints)
            {
                imgpts.push_back(cv::Point2d{cvpoint.x(), cvpoint.y()});
            }
        }
        if (sign.signclass[0] == 'r')
        {
            fivepoints = sign.C; // 中
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() + sign.scale * 0.5 * G / G.norm(); // 右上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() - sign.scale * 0.5 * G / G.norm(); // 右下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() - sign.scale * 0.5 * G / G.norm(); // 左下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm() + sign.scale * 0.5 * G / G.norm(); // 左上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});

            fivepoints = sign.C + sign.scale * 0.5 * G / G.norm(); // 上
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C + 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 左
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - sign.scale * 0.5 * G / G.norm(); // 下
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});
            fivepoints = sign.C - 0.5 * sign.scale * sign.N.cross(G) / (sign.N.cross(G)).norm(); // 右
            realpts.push_back(cv::Point3d{fivepoints.x(), fivepoints.y(), fivepoints.z()});

            for (auto cvpoint : sign.cvPoints)
            {
                imgpts.push_back(cv::Point2d{cvpoint.x(), cvpoint.y()});
            }
        }
    }

    Matrix3d R;
    cv::Mat rot;
    Vector3d T;

    static cv::Mat rvec = (cv::Mat_<double>(3, 1) << 1, 1, 1);
    static cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 2);

    double judge;
    std::cout << "imgpts_number:" << imgpts.size() << std::endl;
    std::cout << "realpts_number:" << realpts.size() << std::endl;
    // for (auto pts : imgpts)
    // {
    //     cout << "cvPoints:" << pts.x << "," << pts.y << endl;
    // }
    // cv::solvePnP(realpts, imgpts, intrinsic, distortion, rvec, tvec, true, cv::SOLVEPNP_EPNP);
    cv::solvePnPRansac(realpts, imgpts,
                       intrinsic, distortion,
                       rvec, tvec, true, 200, 5.0, 0.99);
    cv::projectPoints(realpts, rvec, tvec, intrinsic, distortion, imgpts_);
    std::vector<Vector3d> realpts3d_vec;
    for (int i = 0; i < realpts.size(); i++)
    {
        judge += abs(imgpts_[i].x - imgpts[i].x) + abs(imgpts_[i].y - imgpts[i].y); // 像素投影误差
        Vector3d realpts3d = Vector3d{realpts[i].x, realpts[i].y, realpts[i].z};
        Vector3d realpts3d_ = estimator.ric[0].transpose() * (estimator.Rs[k].transpose() * (realpts3d - estimator.Ps[k]) - estimator.tic[0]);
        realpts3d_vec.push_back(realpts3d_);
    }
    // std::cout << "realpts3d_number:" << realpts3d_vec.size() << std::endl;
    // for (auto pts : realpts3d_vec)
    // {
    //     cout << "ProPoints:" << (pts.x() / pts.z()) * fx + cx << "," << (pts.y() / pts.z()) * fy + cy << endl;
    // }
    cv::Rodrigues(rvec, rot); // 相机到路标系
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R(i, j) = rot.at<double>(i, j);
    for (int i = 0; i < 3; i++)
    {
        int j = 0;
        T(i, j) = tvec.at<double>(i, j);
    }
    cv::Mat axis_pic;
    if (!estimator.img_buf.empty())
    {
        // axis_pic = estimator.img_buf.front().first;
        estimator.img_buf.front().first.clone().copyTo(axis_pic);
        cv::cvtColor(axis_pic, axis_pic, cv::COLOR_BGR2RGB);
        for (int i = 0; i < imgpts_.size(); i++)
        {
            // cv::circle(axis_pic, imgpts_[i], 1, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);                                                                                                                    // b
            cv::circle(axis_pic, imgpts[i], 2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);                                                                                                                     // b
            cv::circle(axis_pic, cv::Point2d((realpts3d_vec[i].x() / realpts3d_vec[i].z()) * fx + cx, (realpts3d_vec[i].y() / realpts3d_vec[i].z()) * fy + cy), 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA); // g
            // cv::line(axis_pic, imgpts_[i], imgpts[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::line(axis_pic, cv::Point2d((realpts3d_vec[i].x() / realpts3d_vec[i].z()) * fx + cx, (realpts3d_vec[i].y() / realpts3d_vec[i].z()) * fy + cy), imgpts[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // r
        }
        // 写入文件
        static int sign_num;
        // string tmp_str;
        // tmp_str = std::to_string(sign_num);
        // char filename[50];
        // sprintf(filename, "/home/scott/gvins_yolo_output/axis-total/%d.jpg", sign_num);
        // cv::imwrite(filename, axis_pic);
        sign_num++;
        // printf("\033[1;35m correct sign! \n\033[0m");
    }
    if (judge < imgpts.size() * 10.0)
    {
        // cout << "R:" << R << endl;                                  // 相机系到世界系
        // cout << "inv_R:" << R.inverse() << endl;                    // 世界系到相机系
        // cout << "real_R" << R * estimator.ric[0].inverse() << endl; // body系到世界系
        // cout << "Rs:" << estimator.Rs[k] << endl;                   // body系到世界系
        // cout << "T:" << T << endl;                                  // 相机系下 相机系到世界系的位移
        // cout << "real_T" << -R.inverse() * T - R * estimator.ric[0].inverse() * estimator.tic[0] << endl;
        // cout << "Ps:" << estimator.Ps[k] << endl; // 相机系在世界系下的位置
        Estimator::RT rt;
        rt.R = R * estimator.ric[0].inverse();
        rt.T = -R.inverse() * T - R * estimator.ric[0].inverse() * estimator.tic[0];

        rt.time = estimator.Headers[k].stamp.toSec();
        estimator.RT_from_signs.push_back(rt);
    }
    return true;
}

// 20230601_xjl
bool processTri(queue<sensor_msgs::PointCloudConstPtr> &tri_buf, Vector3d &c_tri, Vector3d &n_tri, double &time, vector<pair<pair<Eigen::Vector3d, Eigen::Vector3d>, double>> &p_list)
{
    SIGN sign_tri;
    double u, v; // 中心点
    double a1, b1;
    double a2, b2;
    double a3, b3;     // 三顶点
    string classoftri; // 类别
    double similarty;  // 置信度
    double check;
    std::vector<cv::Point3d> srcPoints, dstPoints;
    std::vector<cv::Point2d> pc, draw;
    static std::vector<cv::Point2d> pre_pc;
    static double last_time;
    bool activate;
    double dis;               // 距离
    double true_tri_d = 0.30; // 真实边长30cm
                              //  cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 1765.682901, 0.000000, 782.352086, 0.000000, 1758.799034, 565.999397, 0, 0, 1);
                              //  cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.060942, 0.058542, 0.001478, 0.002002); // camera_matrix(去畸变前的内参)
                              //    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
                              //    cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);
    Matrix3d intrinsic_;
    int is_sign_find = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    double fx, fy, cx, cy, k1, k2, p1, p2;
    double in;
    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);
    k1 = distortion.at<double>(0, 0);
    k2 = distortion.at<double>(0, 1);
    p1 = distortion.at<double>(0, 2);
    p2 = distortion.at<double>(0, 3);

    if (tri_buf.empty())
    {
        return false;
    }
    if (!tri_buf.empty())
    {
        time = tri_buf.front()->header.stamp.toSec();
        classoftri = tri_buf.front()->header.frame_id;

        u = tri_buf.front()->channels[0].values[0]; // u,v 中心点
        v = tri_buf.front()->channels[1].values[0];
        a1 = tri_buf.front()->channels[2].values[0]; // 顶点(按照什么顺序排列需要确定)
        b1 = tri_buf.front()->channels[3].values[0];
        a2 = tri_buf.front()->channels[4].values[0];
        b2 = tri_buf.front()->channels[5].values[0];
        a3 = tri_buf.front()->channels[6].values[0];
        b3 = tri_buf.front()->channels[7].values[0];
        similarty = tri_buf.front()->channels[8].values[0];

        pc.push_back(cv::Point2d(u, v));
        pc.push_back(cv::Point2d(a1, b1));
        pc.push_back(cv::Point2d(a2, b2));
        pc.push_back(cv::Point2d(a3, b3));

        tri_buf.pop();
    }
    while (!estimator.img_buf.empty() && (time > estimator.img_buf.front().second))
    {
        estimator.img_buf.pop_front();
    }
    Vector3d ep, np; // 相机系和世界系下中心点，并求法向量
    std::vector<cv::Point2d> imgpts;
    double judge = 0;
    double judge_ = 0;
    dstPoints.push_back(cv::Point3d(0, 0, 0.01));
    dstPoints.push_back(cv::Point3d(0 - 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0.01));
    dstPoints.push_back(cv::Point3d(0, 0 + true_tri_d * 0.5773502, 0.01)); // 中左下右上
    dstPoints.push_back(cv::Point3d(0 + 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0.01));
    dstPoints.push_back(cv::Point3d(0, 0 - true_tri_d * 0.28867513, 0.01));
    cv::Mat rvec(3, 1, CV_64FC1);
    cv::Mat tvec(3, 1, CV_64FC1); // 相机坐标到路标坐标系的坐标系变换
    // 求出相对相机的坐标
    int index;                                      // 标志对应的滑窗位置索引
    if (!estimator.sign2local(time, index, ep, np)) // 得到标志中心在世界坐标系下坐标
        return false;
    // csv读文件
    Vector3d g;
    // 找到重力对应的投影
    if (!find_gravity(index, g)) // 找到图像对应时间戳下重力g转为相机系下的g
    {
        return false;
    }
    g.normalize();
    vector<Vector2d> uv; // 归一化
    static vector<Vector2d> last_uv;
    std::vector<Vector3d> ps, pw; // 相机系下的坐标,世界系下的坐标
    Vector3d n, pe;               // 两帧计算得出的相机系下的法向量，位置
    Vector3d NN;
    Vector3d CC;
    dis = estimator.computeTri(pc, ep, np, intrinsic_, true_tri_d, index, g); // 估计距离和世界系下的坐标（有误差）
    printf("\033[1;34m c_pos_triangle:%f,%f,%f\n\033[0m", ep.x(), ep.y(), ep.z());
    printf("\033[1;35m n_pos_triangle:%f,%f,%f\n\033[0m", np.x(), np.y(), np.z());
    if (cv::solvePnP(dstPoints, pc, intrinsic, distortion, rvec, tvec, false, cv::SOLVEPNP_EPNP)) // 解P3P
    {
        cv::projectPoints(dstPoints, rvec, tvec, intrinsic, distortion, imgpts);
        for (int i = 0; i < pc.size(); i++)
        {
            judge += abs(imgpts[i].x - pc[i].x) + abs(imgpts[i].y - pc[i].y); // 像素投影误差
        }
    }
    else
        return false;
    int id = -1;
    int id_ = 0;
    estimator.map_manager.addSignCheck(np, classoftri, id); // 在管理器中寻找是否有同一个标志
    CC = np;
    for (int i = 0; i < pc.size(); i++)
    {
        uv.push_back({(pc[i].x - cx) / fx, (pc[i].y - cy) / fy});
    }
    if (id != -1) // 若在管理器中找到了该标志
    {
        printf("find it in map_manager!\n");
        is_sign_find = 1;
        if (!estimator.map_manager.sign.empty())
        {
            for (auto it = estimator.map_manager.sign.begin(), it_next = estimator.map_manager.sign.begin(); it != estimator.map_manager.sign.end(); it = it_next) // 遍历滑窗标志
            {
                if (it->classify == classoftri && ((it->C_ - np).norm() < 2) && id == it->sign_id)
                {
                    if (abs(time - last_time) < 0.05) // if continous
                    {
                        // estimator.computeRT(it->sign_per_frame.back().pts, uv, pe, n);
                        Matrix4d Trw, Tcw;
                        Trw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE - 1] * estimator.ric[0];
                        Trw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE - 1] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE - 1];
                        Trw(3, 3) = 1;
                        Trw(3, 2) = 0;
                        Trw(3, 1) = 0;
                        Trw(3, 0) = 0;
                        // std::cout << Trw << std::endl;;
                        Tcw.block<3, 3>(0, 0) = estimator.Rs[WINDOW_SIZE] * estimator.ric[0];
                        Tcw.block<3, 1>(0, 3) = estimator.Rs[WINDOW_SIZE] * (estimator.ric[0] * estimator.tic[0]) + estimator.Ps[WINDOW_SIZE];
                        Tcw(3, 3) = 1;
                        Tcw(3, 2) = 0;
                        Tcw(3, 1) = 0;
                        Tcw(3, 0) = 0;
                        // std::cout << Tcw << std::endl;;
                        int idx;
                        if (estimator.GetSignTheta(last_uv, uv, pre_pc, pc, Trw, Tcw, idx, n, intrinsic_))
                        {
                            n.normalize();
                            // std::cout << "n:" << n << std::endl;;
                        }
                    }
                    estimator.map_manager.initialSign(id, classoftri, it->C_, it->N_, time, uv, is_sign_find); // 在管理器中初始化一个标志
                    NN = it->N_;
                    NN.normalize();
                    CC = it->C_;
                    break;
                }
                it_next++;
            }
        }
    }
    else // 管理器中没有，去csv文件中找
    {
        if (parseVectorMap(id, np, classoftri, p_list, time, id_)) // 使用id读地图得到所需要的标志的先验信息p_list
        {
            // 此时的id不为-1
            if (!p_list.empty() && id != 1)
            {
                is_sign_find = 2;
                NN = p_list.front().first.second;
                CC = p_list.front().first.first;
                estimator.map_manager.initialSign(id, classoftri, p_list.front().first.first, NN, time, uv, is_sign_find); // 初始化路标到管理器中，只负责加入，和add features一样
                p_list.clear();
            }
        }
    }
    Matrix3d RR_;
    Vector3d t_;
    cv::Mat rot;
    cv::Rodrigues(rvec, rot); // 相机到路标系
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RR_(i, j) = rot.at<double>(i, j);
        }
    }
    t_.x() = tvec.at<double>(0, 0);
    t_.y() = tvec.at<double>(1, 0);
    t_.z() = tvec.at<double>(2, 0);
    // std::cout << "*********************" << t_.x() * (RR_ * Vector3d{0, 0, 1}).x() + t_.y() * (RR_ * Vector3d{0, 0, 1}).y() + t_.z() * (RR_ * Vector3d{0, 0, 1}).z() << std::endl;;
    // std::cout << "t:" << t_ << std::endl;;
    NN = estimator.Rs[index] * estimator.ric[0] * RR_ * (Vector3d{0, 0, 1});
    CC = estimator.Rs[index] * (estimator.ric[0] * t_ + estimator.tic[0]) + estimator.Ps[index];
    printf("\033[1;36m dis of triangle:%f\nc_pos_triangle: %f, %f, %f \nn_pos_triangle:%f,%f,%f\nn_thita_triangle:%f,%f,%f\n\033[0m",
           dis, t_.x(), t_.y(), t_.z(), CC.x(), CC.y(), CC.z(), NN.x(), NN.y(), NN.z());

    vector<Vector3d> worldpoint;
    // // 世界系下的标志投影回到相机系
    worldpoint.push_back(CC);                                                                                                  // 中
    worldpoint.push_back(CC - 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.28867513 * G / G.norm()); // 左
    worldpoint.push_back(CC - true_tri_d * 0.5773502 * G / G.norm());                                                          // 下
    worldpoint.push_back(CC + 0.5 * true_tri_d * NN.cross(G) / (NN.cross(G)).norm() + true_tri_d * 0.28867513 * G / G.norm()); // 右
    worldpoint.push_back(CC + true_tri_d * 0.28867513 * G / G.norm());                                                         // 上
    for (auto wp : worldpoint)
    {
        int i = 0;
        Vector3d gg = estimator.ric[0].transpose() * (estimator.Rs[index].transpose() * (wp - estimator.Ps[index]) - estimator.tic[0]);
        cv::Point3d g;
        if (gg.z() < 0)
            return false;
        // gg = intrinsic_ * gg;
        g.x = gg.x() / gg.z();
        g.y = gg.y() / gg.z();
        g.z = gg.z() / gg.z();
        draw.push_back(cv::Point2d(g.x, g.y));
        judge_ += abs(uv[i].x() - g.x) + abs(uv[i].y() - g.y);
        i++;
    }
    std::cout << "judge_" << judge_ << std::endl;
    ;
    // 中左下右
    // std::cout << (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 0} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index]) << std::endl;;
    // std::cout << (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, true_tri_d * 0.5773502, 0} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index]) << std::endl;;
    // judge_ += (CC - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, 0, 0.01} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[1] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0 - 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (CC - true_tri_d * G / G.norm() - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0, true_tri_d * 0.5773502, 0.01} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // judge_ += (pw[3] - (estimator.Rs[index] * (estimator.ric[0] * (RR_ * Vector3d{0 + 0.5 * true_tri_d, 0 - true_tri_d * 0.28867513, 0} + Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))) + estimator.tic[0]) + estimator.Ps[index])).norm();
    // if (judge < 10 && judge_ < 0.5 && dis < 8 && tvec.at<double>(0, 2) > 0)
    if (judge < 10 && dis < 8 && tvec.at<double>(0, 2) > 0)
    {
        vector<cv::Point2d> input_pts1, input_pts2, input_pts3;
        input_pts1.push_back(pc[0]);
        input_pts1.push_back(pc[1]);
        input_pts1.push_back(pc[3]);
        input_pts2.push_back(draw[0]);
        input_pts2.push_back(draw[1]);
        input_pts2.push_back(draw[3]);
        input_pts3.push_back(imgpts[0]);
        input_pts3.push_back(imgpts[1]);
        input_pts3.push_back(imgpts[3]);
        if (!estimator.choose_right(input_pts1, input_pts2) || !estimator.choose_right(input_pts1, input_pts3))
        {
            std::cout << "wrong triangle answer!!!" << std::endl;
            return false;
        }
        else
        {
            std::cout << "right triangle answer!!!" << std::endl;
        }
        // 初始化标志到标志管理器和csv
        Matrix3d Rsw;
        Rsw = (estimator.Rs[index] * estimator.ric[0] * RR_).transpose();
        // std::cout << (estimator.Rs[index] * estimator.ric[0] * RR_).transpose() << std::endl;;
        SIGN temp_sign;
        temp_sign.signclass = classoftri;
        temp_sign.C = CC;
        NN.z() = 0;
        NN.normalize();
        temp_sign.N = estimator.Rs[index] * estimator.ric[0] * RR_ * Vector3d{0, 0, 1};
        temp_sign.q = Quaterniond(estimator.Rs[index] * estimator.ric[0] * RR_);
        temp_sign.cvPoints.clear();
        for (int i = 0; i < pc.size(); i++)
        {
            temp_sign.cvPoints.push_back(Vector2d{uv[i].x(), uv[i].y()});
        }
        temp_sign.ric = estimator.ric[0];
        temp_sign.tic = estimator.tic[0];
        temp_sign.dis = dis;
        temp_sign.similarty = similarty;
        temp_sign.scale = 0.15;
        estimator.mapforsign.push_back(temp_sign);
        std::cout << "add triagnle!!!" << std::endl;
        ;
        if (id == -1)
        {
            // 计算在世界系下的法向量和中心点
            // estimator.map_manager.initialSign(id_, classoftri, CC, NN, time, uv, is_sign_find); // 初始化路标
            // estimator.para_sign_Pose[id_][0] = CC.x();                                          // 中心
            // estimator.para_sign_Pose[id_][1] = CC.y();
            // estimator.para_sign_Pose[id_][2] = CC.z();
            // estimator.para_sign_Pose[id_][3] = NN.x(); // 法向量
            // estimator.para_sign_Pose[id_][4] = NN.y();
            // estimator.para_sign_Pose[id_][5] = NN.z();

            //     printf("\033[1;32m No Data Association!\n\033[0m");
            //     {
            //         ofstream f("/home/seu/xjl_work_space/gvins_yolo_ws/src/VINS-Mono-master/map/map.csv", ios::app);
            //         f.setf(ios::fixed, ios::floatfield);
            //         if (!f)
            //         {
            //             std::cout << "打开失败！请重试！" << std::endl;;
            //             return false;
            //         }
            //         else
            //         {
            //             f.precision(9);
            //             f << id_ << ","
            //               << classoftri << ","
            //               << CC[0] << ","
            //               << CC[1] << ","
            //               << CC[2] << ","
            //               << NN.x() << ","
            //               << NN.y() << ","
            //               << NN.z()
            //               << ","
            //               << "0.30"
            //               << ","
            //               << time << std::endl;;
            //             f.close();
            //             printf("\033[1;35m 写入三角形标志数据!\n\033[0m");
            //         }
            //     }
        }
        // cv::Mat axis_pic;
        // if (!estimator.img_buf.empty())
        // {
        // axis_pic = estimator.img_buf.front().first;
        // estimator.img_buf.front().first.clone().copyTo(axis_pic);

        // for (int i = 0; i < draw.size(); i++)
        // {
        //     draw[i].x = draw[i].x * fx + cx;
        //     draw[i].y = draw[i].y * fy + cy;
        //     cv::circle(axis_pic, draw[i], 3, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        //     cv::circle(axis_pic, pc[i], 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        //     cv::line(axis_pic, draw[i], pc[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        // }
        // // 写入文件
        // static int t_num;
        // string tmp_str;
        // tmp_str = std::to_string(t_num);
        // char filename[50];
        // sprintf(filename, "/home/scott/gvins_yolo_output/axis-tri/%d.jpg", t_num);
        // cv::imwrite(filename, axis_pic);
        // t_num++;
        // printf("\033[1;35m correct triangle! \n\033[0m");
        // }
    }
    else
    {
        return false;
    }

    last_time = time;
    pre_pc = pc;
    last_uv = uv;
    draw.clear();
    return true;
}

// 20230601_xjl
bool processRect(queue<sensor_msgs::PointCloudConstPtr> &rect_buf, Vector3d &rect, double &time, cv::Mat3d &H, vector<pair<pair<cv::Point3d, double>, double>> &p_list)
{
    double u, v;
    double a1, b1;
    double a2, b2;
    double a3, b3;
    double a4, b4;
    string class_of_rect;
    // double ori;
    // double time;
    int id;
    double dis;
    // double true_rect_d = 0.1; //真实半径10cm
    double true_rect_d; // 真实半径
                        // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
                        // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
                        // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 1765.682901, 0.000000, 782.352086, 0.000000, 1758.799034, 565.999397, 0, 0, 1);
                        // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.060942, 0.058542, 0.001478, 0.002002);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);
    Vector3d p;
    // Vector3d p1;
    // Vector3d p2;
    // Vector3d p3;
    // Vector3d p4;
    cv::Point2f p1_, p11;
    cv::Point2f p2_, p22;
    cv::Point2f p3_, p33;
    cv::Point2f p4_, p44;
    std::vector<cv::Point2f> srcPoints, dstPoints;
    double fx, fy, cx, cy;
    double in;

    fx = intrinsic.at<double>(0, 0);
    fy = intrinsic.at<double>(1, 1);
    cx = intrinsic.at<double>(0, 2);
    cy = intrinsic.at<double>(1, 2);

    if (rect_buf.empty())
    {
        return false;
    }
    if (!rect_buf.empty())
    {
        time = rect_buf.front()->header.stamp.toSec();
        class_of_rect = rect_buf.front()->header.frame_id;

        // u = ellipse_buf.front()->channels();
        p.x() = rect_buf.front()->points[0].x; // 去畸变后的中心点坐标
        p.y() = rect_buf.front()->points[0].y;
        p.z() = rect_buf.front()->points[0].z; // z=1，归一化

        u = rect_buf.front()->channels[0].values[0]; // u,v 中心点
        v = rect_buf.front()->channels[1].values[0];

        a1 = rect_buf.front()->channels[2].values[0]; // 顶点
        b1 = rect_buf.front()->channels[3].values[0];
        // p1_.x = a1;
        // p1_.y = b1;

        // p1.x() = a1; // 去畸变后的坐标
        // p1.y() = b1;
        // p1.z() = 1; // z=1，归一化

        a2 = rect_buf.front()->channels[4].values[0];
        b2 = rect_buf.front()->channels[5].values[0];
        // p2_.x = a2;
        // p2_.y = b2;

        // p2.x() = a2; // 去畸变后的坐标
        // p2.y() = b2;
        // p2.z() = 1; // z=1，归一化

        a3 = rect_buf.front()->channels[6].values[0];
        b3 = rect_buf.front()->channels[7].values[0];
        // p3_.x = a3;
        // p3_.y = b3;

        // p3.x() = a3; // 去畸变后的坐标
        // p3.y() = b3;
        // p3.z() = 1; // z=1，归一化

        a4 = rect_buf.front()->channels[8].values[0];
        b4 = rect_buf.front()->channels[9].values[0];
        // p4_.x = a4;
        // p4_.y = b4;

        // p4.x() = a4; // 去畸变后的坐标
        // p4.y() = b4;
        // p4.z() = 1; // z=1，归一化

        srcPoints.push_back(cv::Point2f(u, v));
        srcPoints.push_back(p1_);
        srcPoints.push_back(p2_);
        srcPoints.push_back(p3_);
        srcPoints.push_back(p4_);
        rect_buf.pop();
    }
    in = sqrt(((u - cx) * (u - cx)) / (fx * fx) + ((v - cy) * (v - cy)) / (fy * fy) + 0.0001);
    double least_ = 3;
    pair<pair<cv::Point3d, double>, double> pp;
    // 选出距离最近的那个标志信息
    if (p_list.size() != 0)
    {
        for (auto p : p_list)
        {
            double dis_ = (rect.x() - p.first.first.x) * (rect.x() - p.first.first.x) + (rect.y() - p.first.first.y) * (rect.y() - p.first.first.y) + (rect.z() - p.first.first.z) * (rect.z() - p.first.first.z);
            if (least_ > dis)
            {
                least_ = dis;
                pp = p;
            }
            else
                continue;
        }
    }
    p_list.clear();
    p_list.push_back(pp);
    // 坐标
    pp.first.first.x;
    pp.first.first.y;
    pp.first.first.z;
    true_rect_d = pp.second;
    // pp.first.second;//航向角
    dstPoints.push_back(cv::Point2f(0, 0));
    dstPoints.push_back(cv::Point2f(0 + 0.5 * true_rect_d, 0 + true_rect_d * 0.5));
    dstPoints.push_back(cv::Point2f(0 + 0.5 * true_rect_d, 0 - true_rect_d * 0.5));
    dstPoints.push_back(cv::Point2f(0 - 0.5 * true_rect_d, 0 - true_rect_d * 0.5));
    dstPoints.push_back(cv::Point2f(0 - 0.5 * true_rect_d, 0 + true_rect_d * 0.5));
    // 已知图像中最外围的四个角点的世界坐标（x,y,z）.估计两坐标系之间的单应矩阵H
    H = cv::findHomography(srcPoints, dstPoints);

    // dis = (in * true_ellipse_d * (fx + fy) / 2) / a; //距离

    double n = dis / in;

    rect.x() = (u - cx) / fx * n;
    rect.y() = 1 * n;
    rect.z() = -(v - cy) / fy * n;
    return true;
}

bool processArUco(queue<MARKER> &aruco_buf, Vector3d &ep, Vector3d &np, Quaterniond &eq, Quaterniond &nq, double &time)
{
    Eigen::Matrix3d R;
    if (!aruco_buf.empty())
    {
        time = aruco_buf.front().time;
        int idofmarker = aruco_buf.front().id;
        ep.x() = aruco_buf.front().pose.x();
        ep.y() = aruco_buf.front().pose.y();
        ep.z() = aruco_buf.front().pose.z();
        eq.w() = aruco_buf.front().ori.w();
        eq.x() = aruco_buf.front().ori.x();
        eq.y() = aruco_buf.front().ori.y();
        eq.z() = aruco_buf.front().ori.z();
    }
    else
    {
        return false;
    }
    while (!aruco_buf.empty() && time < estimator.Headers[0].stamp.toSec())
    {
        aruco_buf.pop();
    }
    if (time >= estimator.Headers[0].stamp.toSec() && time <= estimator.Headers[WINDOW_SIZE - 1].stamp.toSec())
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            if (time == estimator.Headers[i].stamp.toSec())
            {
                np = estimator.Rs[i] * (estimator.ric[0] * ep + estimator.tic[0]) + estimator.Ps[i]; // 世界系下坐标 //标志系到世界系姿态
                R = estimator.Rs[i] * estimator.ric[0] * eq.toRotationMatrix();
                nq = Quaterniond(R);
                aruco_buf.pop();
                return true;
            }
            else
                continue;
        }
    }
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // imu最后一帧的时间不比视觉的第一帧大，需要等一会imu
        // imu                           ********
        // feature                                        *    *    *
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu第一帧的时间比视觉的第一帧靠后，需要丢几个视觉帧
        // imu                           ********
        // feature                 *    *    *
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        // 剩下的情况就是
        //  imu                  ***********
        //  feature                 *    *    *
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        // 当imu的时间戳小于视觉时间戳就塞入队列中
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 再塞一帧在视觉时间之后的imu好做插值
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        // 形如
        //  imu                  ****
        //  feature                 *
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }

    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    // 唤醒的是process线程
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        // 从IMU测量值imu_msg和上一个PVQ递推得出下一个的PVQ
        predict(imu_msg); // 预积分
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // 发布最新的由predict得到的 P V Q
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布pvq信息
    }
}

void gnss_callback(const sensor_msgs::NavSatFixConstPtr &gnssmsg)
{
    // Time convertion
    // double unixsecond = gnssmsg->header.stamp.toSec();
    double weeksec;
    // int week;

    // GpsTime::unix2gps(unixsecond, week, weeksec);
    weeksec = gnssmsg->header.stamp.toSec() - 0.02;
    // weeksec = gnssmsg->header.stamp.toSec();

    static double last_gnss_100HZ = 0.0;
    if (GINS_INTIALING == false)
    {
        if (weeksec - last_gnss_100HZ <= 0.99)
        {
            return;
        }
        last_gnss_100HZ = weeksec;
    }
    // else
    // {
    //     if (weeksec - last_gnss_100HZ <= 0.15)
    //     {
    //         return;
    //     }
    //     last_gnss_100HZ = weeksec;
    // }
    // if (weeksec - last_gnss_100HZ <= 0.99)
    // {
    //     return;
    // }
    // last_gnss_100HZ = weeksec;

    // GNSS gnss_ = {0.0, Vector3d{0, 0, 0}, Vector3d{0, 0, 0}, false, 0.0};
    GNSS gnss_;
    gnss_.time = weeksec;

    // gnss_.blh[0] = gnssmsg->latitude * D2R;  // 纬度
    // gnss_.blh[1] = gnssmsg->longitude * D2R; // 经度
    gnss_.blh[0] = gnssmsg->latitude;  // 纬度
    gnss_.blh[1] = gnssmsg->longitude; // 经度
    gnss_.blh[2] = gnssmsg->altitude;  // 高程
    // gnss_.std[0] = (gnssmsg->position_covariance[4]); // N
    // gnss_.std[1] = (gnssmsg->position_covariance[0]); // E
    // gnss_.std[2] = (gnssmsg->position_covariance[8]); // D
    gnss_.std[0] = 0.50;     // N
    gnss_.std[1] = 0.50;     // E
    gnss_.std[2] = 0.50; // D

    gnss_.isyawvalid = false;

    // if ((gnss_.std[0] == 0) || (gnss_.std[1] == 0) || (gnss_.std[2] == 0))
    // {
    //     return;
    // }
    // Remove bad GNSS
    bool isoutage = false;
    // if ((gnss_.std[0] < GNSS_THRESHOLD) && (gnss_.std[1] < GNSS_THRESHOLD) && (gnss_.std[2] < GNSS_THRESHOLD))
    // {
    if (origin.isZero())
    {
        origin = gnss_.blh;
        std::cout << "origin  : " << std::endl;
        ;
        printf("%.9f \n", origin[0] * R2D);
        printf("%.9f \n", origin[1] * R2D);
        printf("%.9f \n", origin[2]);
        // std::cout<<origin[2]<<std::endl;;
    }
    G.z() = Earth::gravity(gnss_.blh);
    Vector3d blh;
    blh = Earth::global2local(origin, gnss_.blh); // 换算为东北天坐标系下的坐标
    // if (!blh.isZero())
    // {
    // std::cout << "b " << std::endl;;
    // std::cout << blh << std::endl;;
    // return;
    // }
    /*******************************************/
    // 20230907_xjl
    // Vector3d carblh, signblh;
    // carblh[0] = (32.0 + 3.0 / 60.0 + 28.43391 / 3600.0) * D2R;
    // carblh[1] = (118.0 + 47.0 / 60.0 + 3.98469 / 3600.0) * D2R;
    // carblh[2] = (10.784 - 1.6 + 1.2);
    // signblh[0] = (32.0 + 3.0 / 60.0 + 28.60009 / 3600.0) * D2R;
    // signblh[1] = (118.0 + 47.0 / 60.0 + 4.12855 / 3600.0) * D2R;
    // signblh[2] = (10.796 - 1.6 + 0.15);
    //  carblh.x0.347744766
    //  carblh.y-0.683260
    //  carblh.z0.701000
    //  signblh.x5.466458
    //  signblh.y3.090420
    //  signblh.z1.039003
    //  0718_02bag包

    // 32.057849923
    // 118.784445652
    // 10.088000000
    // Vector3d aa = Earth::global2local(origin, carblh);
    // Vector3d bb = Earth::global2local(origin, signblh);
    // printf("carblh.x%.9f\n", aa[0]);
    // printf("carblh.y%f\n", aa[1]);
    // printf("carblh.z%f\n", aa[2]);
    // printf("signblh.x%f\n", bb[0]);
    // printf("signblh.y%f\n", bb[1]);
    // printf("signblh.z%f\n", bb[2]);
    /******************************************/

    //     if (isusegnssoutage_ && (weeksec >= gnssoutagetime_)) {
    //         isoutage = true;
    //     }

    gnss_.blh[0] = blh[1];
    gnss_.blh[1] = blh[0];
    gnss_.blh[2] = -blh[2];
    // 低频不加锁，高频就加
    // m_buf.lock();
    gnss_buf.push(gnss_);
    // m_buf.unlock();
    // 20221209_xjl write gnss
    {
        ofstream foutC("/home/scott/gvins_yolo_output/gnss_result.txt", ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(9);
        foutC << gnss_.time << " ";
        foutC.precision(5);
        foutC << gnss_.blh[0] << " "
              << gnss_.blh[1] << " "
              << gnss_.blh[2] << " "
              // ecef
              // foutC << ecef(0) << " "
              //       << ecef(1) << " "
              //       << ecef(2) << " "
              << "0"
              << " "
              << "0"
              << " "
              << "0"
              << " "
              << "1" << std::endl;
        ;
        foutC.close();
    }
    return;

    //     // add new GNSS to GVINS
    //     if (!isoutage) {
    //         gvins_->addNewGnss(gnss_);
    //     }
    // }
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    // std::cout << "points number : " << feature_msg->points.size() << std::endl;;
    if (!init_feature)
    {
        // skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void ellipse_callback(const sensor_msgs::PointCloudConstPtr &ellipse_msg) // 处理路标信息
{
    // m_buf.lock();
    ellipse_buf.push(ellipse_msg);
    // m_buf.unlock();
}

void tri_callback(const sensor_msgs::PointCloudConstPtr &tri_msg) // 处理路标信息
{
    tri_buf.push(tri_msg);
    // SIGN sign_tri;
    // sign_tri.time=tri_msg->header.stamp.toSec();
    // sign_buf.push(sign_tri);
}

void rect_callback(const sensor_msgs::PointCloudConstPtr &rect_msg) // 处理路标信息
{
    // if (rect_msg->header.frame_id[0] == 'c')
    //     std::cout << "cir!!!" << std::endl;;
    // if (rect_msg->header.frame_id[0] == 't')
    //     std::cout << "tri!!!" << std::endl;;
    // if (rect_msg->header.frame_id[0] == 'r')
    //     std::cout << "rect!!!" << std::endl;;
    // rect_buf.push(rect_msg);
    // SIGN sign_rect;
    // sign_rect.time=rect_msg->header.stamp.toSec();
    // sign_rect.xyz.x();
    rect_buf.push(rect_msg);
    // std::cout << "points number : " << rect_msg->points.size() << std::endl;;
}

void sign_callback(const sensor_msgs::PointCloudConstPtr &sign_msg) // 处理路标信息
{
    // std::cout << "points number : " << sign_msg->points.size() << std::endl;;
    pair<double, std::vector<SIGN>> signvec;
    signvec.first = sign_msg->header.stamp.toSec();
    // printf("pointcloud_time:%f\n", signvec.first);
    // std::cout << "************************" << std::endl;;
    if (!sign_msg->points.empty())
    {
        for (unsigned int i = 0; i < sign_msg->points.size(); i++)
        {
            SIGN temp_sign;
            // image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            // double id = sign_msg->channels[12 * i + 0].values[i];
            double id = sign_msg->channels[0].values[i];
            // printf("id:%f", id);
            temp_sign.signclass = sign_id_to_class(id);
            // std::cout << "sign class : " << temp_sign.signclass << std::endl;;
            temp_sign.C.x() = sign_msg->points[i].x;
            temp_sign.C.y() = sign_msg->points[i].y;
            temp_sign.C.z() = sign_msg->points[i].z;
            temp_sign.time = sign_msg->header.stamp.toSec();
            if (temp_sign.signclass[0] == 'c')
            {
                temp_sign.features.push_back(sign_msg->channels[1].values[i]); // u
                temp_sign.features.push_back(sign_msg->channels[2].values[i]); // v
                temp_sign.features.push_back(sign_msg->channels[3].values[i]); // a
                temp_sign.features.push_back(sign_msg->channels[4].values[i]); // b
                temp_sign.features.push_back(sign_msg->channels[5].values[i]); // ori

                temp_sign.features.push_back(sign_msg->channels[6].values[i]);  // u
                temp_sign.features.push_back(sign_msg->channels[7].values[i]);  // v
                temp_sign.features.push_back(sign_msg->channels[8].values[i]);  // a
                temp_sign.features.push_back(sign_msg->channels[9].values[i]);  // b
                temp_sign.features.push_back(sign_msg->channels[10].values[i]); // ori

                temp_sign.features.push_back(sign_msg->channels[11].values[i]); // sim
                temp_sign.similarty = sign_msg->channels[11].values[i];
            }
            else if (temp_sign.signclass[0] == 't')
            {
                temp_sign.features.push_back(sign_msg->channels[1].values[i]);  // u0
                temp_sign.features.push_back(sign_msg->channels[2].values[i]);  // v0
                temp_sign.features.push_back(sign_msg->channels[3].values[i]);  // u1
                temp_sign.features.push_back(sign_msg->channels[4].values[i]);  // v1
                temp_sign.features.push_back(sign_msg->channels[5].values[i]);  // u2
                temp_sign.features.push_back(sign_msg->channels[6].values[i]);  // v2
                temp_sign.features.push_back(sign_msg->channels[7].values[i]);  // u3
                temp_sign.features.push_back(sign_msg->channels[8].values[i]);  // v3
                temp_sign.features.push_back(sign_msg->channels[9].values[i]);  // u4
                temp_sign.features.push_back(sign_msg->channels[10].values[i]); // v4
                temp_sign.features.push_back(sign_msg->channels[11].values[i]); // sim
                temp_sign.similarty = sign_msg->channels[11].values[i];
            }
            else if (temp_sign.signclass[0] == 'r')
            {
                temp_sign.features.push_back(sign_msg->channels[1].values[i]);  // u0
                temp_sign.features.push_back(sign_msg->channels[2].values[i]);  // v0
                temp_sign.features.push_back(sign_msg->channels[3].values[i]);  // u1
                temp_sign.features.push_back(sign_msg->channels[4].values[i]);  // v1
                temp_sign.features.push_back(sign_msg->channels[5].values[i]);  // u2
                temp_sign.features.push_back(sign_msg->channels[6].values[i]);  // v2
                temp_sign.features.push_back(sign_msg->channels[7].values[i]);  // u3
                temp_sign.features.push_back(sign_msg->channels[8].values[i]);  // v3
                temp_sign.features.push_back(sign_msg->channels[9].values[i]);  // u4
                temp_sign.features.push_back(sign_msg->channels[10].values[i]); // v4
                temp_sign.features.push_back(sign_msg->channels[11].values[i]); // sim
                // for (auto fea : temp_sign.features)
                // {
                //     std::cout << "fea" << fea << std::endl;;
                // }
                temp_sign.similarty = sign_msg->channels[11].values[i];
            }
            signvec.second.push_back(temp_sign);
        }
        sign_queue.push(signvec);
    }
}

void aruco_callback(const geometry_msgs::PoseStampedConstPtr &marker_msg)
{
    MARKER marker;
    if (marker_msg->header.stamp.toSec() > 0)
    {
        marker.time = marker_msg->header.stamp.toSec();
        marker.pose.x() = marker_msg->pose.position.x;
        marker.pose.y() = marker_msg->pose.position.y;
        marker.pose.z() = marker_msg->pose.position.z;
        marker.ori.w() = marker_msg->pose.orientation.w;
        marker.ori.x() = marker_msg->pose.orientation.x;
        marker.ori.y() = marker_msg->pose.orientation.y;
        marker.ori.z() = marker_msg->pose.orientation.z;
    }
    aruco_buf.push(marker);
}

void map_callback(const sensor_msgs::NavSatFixConstPtr &map_msg)
{
    double weeksec;
    // int week;

    // GpsTime::unix2gps(unixsecond, week, weeksec);
    weeksec = map_msg->header.stamp.toSec();
    // weeksec = gnssmsg->header.stamp.toSec();

    static double last_map_100HZ = 0.0;
    if (GINS_INTIALING == false)
    {
        if (weeksec - last_map_100HZ <= 0.99)
        {
            return;
        }
        last_map_100HZ = weeksec;
    }
    // else
    // {
    //     if (weeksec - last_gnss_100HZ <= 0.15)
    //     {
    //         return;
    //     }
    //     last_gnss_100HZ = weeksec;
    // }
    // if (weeksec - last_gnss_100HZ <= 0.99)
    // {
    //     return;
    // }
    // last_gnss_100HZ = weeksec;

    // GNSS map_ = {0.0, Vector3d{0, 0, 0}, Vector3d{0, 0, 0}, false, 0.0};
    GNSS map_;
    map_.time = weeksec;

    // map_.blh[0] = map_msg->latitude * D2R;  // 经度
    // map_.blh[1] = map_msg->longitude * D2R; // 纬度
    map_.blh[0] = map_msg->latitude;  // 纬度
    map_.blh[1] = map_msg->longitude; // 经度
    map_.blh[2] = map_msg->altitude;  // 高程
    // map_.blh[0] = map_msg->x;  // 纬度
    // map_.blh[1] = map_msg->y; // 经度
    // map_.blh[2] = map_msg->z;  // 高程
    // map_.std[0] = (map_msg->position_covariance[4]); // N
    // map_.std[1] = (map_msg->position_covariance[0]); // E
    // map_.std[2] = (map_msg->position_covariance[8]); // D
    // map_.std[0] = 0.1; // N
    // map_.std[1] = 0.1; // E
    // map_.std[2] = 0.1; // D
    // map_.std[0] = map_msg->position_covariance[3]; // yaw
    map_.std[0] = 0.05; // yaw
    map_.std[1] = 0.05;
    map_.std[2] = 0.01;
    map_.isyawvalid = false;

    // map_.isyawvalid = true;
    bool isoutage = false;
    // if ((gnss_.std[0] < GNSS_THRESHOLD) && (gnss_.std[1] < GNSS_THRESHOLD) && (gnss_.std[2] < GNSS_THRESHOLD))
    // {
    if (origin.isZero())
    {
        origin = map_.blh;
        std::cout << "origin  : " << std::endl;
        ;
        printf("%.9f \n", origin[0] * R2D);
        printf("%.9f \n", origin[1] * R2D);
        printf("%.9f \n", origin[2]);
        // std::cout<<origin[2]<<std::endl;;
    }
    // G.z() = Earth::gravity(map_.blh);
    // Vector3d blh;
    // blh = Earth::global2local(origin, map_.blh); // 换算为北东地坐标系下的坐标
    // Vector3d blh;
    // blh = Earth::global2local(origin, map_.blh); // 换算为北东地坐标系下的坐标
    // 转为东北天
    // map_.blh[0] = blh[1];
    // map_.blh[1] = blh[0];
    // map_.blh[2] = -blh[2];
    // 低频不加锁，高频就加
    // m_buf.lock();
    gnss_buf.push(map_);
    // map_buf.push(map_);
    // m_buf.unlock();
    // 20221209_xjl write gnss
    {
        ofstream foutC("/home/scott/gvins_yolo_output/map_result.txt", ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(9);
        foutC << map_.time << " ";
        foutC.precision(5);
        foutC << map_.blh[0] << " "
              << map_.blh[1] << " "
              << map_.blh[2] << " "
              // ecef
              // foutC << ecef(0) << " "
              //       << ecef(1) << " "
              //       << ecef(2) << " "
              // yaw
              << "0"
              << " "
              << "0"
              << " "
              << "0"
              << " "
              << "1" << std::endl;
        ;
        foutC.close();
    }
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    // printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// 20221105xjl
// 返回当前timelist中与输入time相等的第cnt个时间戳t
int find_timeindex(double time, std::deque<double> &timelist)
{
    double t;
    int cnt = 0;
    while (!timelist.empty())
    {
        t = timelist.front();
        if (time < t)
        {
            return cnt;
        }
        else
        {
            cnt++;
            timelist.pop_front();
            // t=timelist.front();
        }
    }
}

// 20221119xjl
bool find_gnss(double t, double &time)
{
    while (!gnss_buf.empty())
    {
        time = gnss_buf.front().time;
        if (time < t)
        {
            gnss_buf.pop();
            continue;
        }
        if (time - t < 0.01 || time == t)
        {
            return true;
        }
        return false;
    }
    return false;
}

// 线程入口函数20240906xjl
// void threadFunction(SIGN sign_in_one_frame, const cv::Mat &intrinsic, const cv::Mat &distortion, int k, int &used_sign_number, vector<SIGN> &used_sign)
// {
//     {
//         if (sign_in_one_frame.signclass[0] == 'c')
//         {
//             if (process_Ellipse(sign_in_one_frame, intrinsic, distortion, k))
//             {
//                 printf("cir sign %s detected!\n", sign_in_one_frame.signclass.c_str());
//                 used_sign_number++;
//                 used_sign.push_back(sign_in_one_frame);
//             }
//         }
//         else if (sign_in_one_frame.signclass[0] == 't')
//         {
//             if (process_Tri(sign_in_one_frame, intrinsic, distortion, k))
//             {
//                 printf("tri sign %s detected!\n", sign_in_one_frame.signclass.c_str());
//                 used_sign_number++;
//                 used_sign.push_back(sign_in_one_frame);
//             }
//         }
//         else if (sign_in_one_frame.signclass[0] == 'r')
//         {
//             if (process_Rect(sign_in_one_frame, intrinsic, distortion, k))
//             {
//                 printf("rect sign %s detected!\n", sign_in_one_frame.signclass.c_str());
//                 used_sign_number++;
//                 used_sign.push_back(sign_in_one_frame);
//             }
//         }
//     }
// }



// thread: visual-inertial odometry
void process()
{
    if(gnss_buf.empty())
    {
        estimator.groundtruth = estimator.loadOdometryFromTumFile(GROUNDTRUTH);
        for(auto gt:estimator.groundtruth)
        {
            GNSS map;
            map.time = gt.header.stamp.toSec();
            map.blh[0]=gt.pose.pose.position.x;
            map.blh[1]=gt.pose.pose.position.y;
            map.blh[2]=gt.pose.pose.position.z;
            gnss_buf.push(map);
        }
    }
    while (true)
    {

        // 20221022xjl gins初始化：零速度检测+初始化东北天坐标系下位置及姿态
        //  std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, GNSS>> gins_measurements;
        // 会弹出队列中的第一个gnss以及它之前的imu数据

        // if (GINS_INTIALING == false)
        // {
        //     if ((!imu_buf.empty()) && (!gnss_buf.empty()))
        //     {
        //         m_buf.lock();
        //         GINS_INTIALING = gvinsInitialization();
        //         // std::cout<<GINS_INTIALING<<std::endl;;
        //         m_buf.unlock();
        //     }
        //     continue;
        // }

        // 声明打包好的IMU和图像关键帧容器
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
                    //等待数据接收完成会被唤醒，在执行getMeasurement()提取measurements时互斥锁m_buf会锁住，此时无法接收数据
                    //若measurement序列不为空则继续运行，若为空则程序停在这里
                    //返回的是形如
                    //imu *****
                    //img         *
                    //数据，需要有两帧imu夹住img帧
            return (measurements = getMeasurements()).size() != 0; });
        // 解锁
        lk.unlock();
        // 打开状态估计器
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            // std::cout<<measurement.first.size()<<std::endl;;
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历measurement里的imu序例，对每一段都做预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();                    // imu时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 特征点时间戳

                if (t <= img_t) // 若imu帧的时间戳在img帧时间戳前，此种情况是imu和imu帧之间的积分，也有img到imu的积分
                // 形如
                //  imu    * * * * * * *
                //  img      *             *          ，可以看到开始的第一个预积分是img到imu，中间部分是imu到imu
                {
                    if (current_time < 0) // 初始化是-1
                        current_time = t;
                    double dt = t - current_time; // 初始化就是0，否则是上一帧和这一帧的时差
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // 对每个打包数据进行imu预积分，其中就对estimator的acc和ang_vel进行了更新，Ps和Rs也更新了
                    // printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else // imu帧在img帧之后，此种情况是imu积分到img时刻为止，需要做插值
                {
                    double dt_1 = img_t - current_time; // 当前img帧和上一帧imu的时差
                    double dt_2 = t - img_t;            // 当前imu帧和img帧的时差
                    current_time = img_t;               // 将img时间戳给current_time
                    // 判断正负
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 做了一个插值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // 到插值时刻的预积分，acc、ang_vel、Ps和Rs更新
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty()) // relo_buf不为空
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL) // relo_msg不为空
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // 参考帧属性，初始化选取一个参考帧
                //  获取平移向量
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]); // t
                // 获取旋转四元数
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]); // q
                Matrix3d relo_r = relo_q.toRotationMatrix();                                                                                                            // R
                int frame_index;
                // 匹配的特征点所在的id
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            // 特征点数据结构：<特征点id1，vector<相机id 1<归一化平面坐标3，像素坐标2，像素速度2>>>
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            std::vector<SIGN> used_sign;
            // 遍历这帧的特征点，都是和上一帧匹配对应的特征点
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                // 特征点id
                int feature_id = v / NUM_OF_CAM;
                // 相机id  单目一般是0
                int camera_id = v % NUM_OF_CAM;
                // 地图点坐标
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                // 像素坐标
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                // 像素速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                // 判断深度是否是1
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }
            // 20231129_xjl
            bool keyframe = false;
            if (MAP_INTIALING == false)
            {
                if (MAP_INTIALING = InitialVectorMap(estimator))
                    ROS_INFO("MAP INTIALING!!!");
            }
            // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
            // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.049547, 0.012867, -0.000750, -0.000176);

            // std::vector<std::thread> threads;
            while (!sign_queue.empty() && sign_queue.front().first < estimator.Headers[0].stamp.toSec())
            {
                // printf("pop:%f,img_time[0]:%f\n", sign_queue.front().first, estimator.Headers[0].stamp.toSec());
                sign_queue.pop();
            }
            estimator.multi_sign = false;
            int used_sign_number = 0;
            while (!sign_queue.empty())
            {
                pair<double, vector<SIGN>> signs_in_one_frame = sign_queue.front();
                if (signs_in_one_frame.first == img_msg->header.stamp.toSec()) // 只处理到当前帧之前一帧
                {
                    break;
                }
                sign_queue.pop();
                while (!estimator.img_buf.empty() && estimator.img_buf.front().second < signs_in_one_frame.first)
                {
                    estimator.img_buf.pop_front();
                }
                for (int k = 0; k < WINDOW_SIZE; k++)
                {
                    if (estimator.isTheSameTimeNode(signs_in_one_frame.first, estimator.Headers[k].stamp.toSec(), 0.01) && estimator.solver_flag > 0)
                    {
                        printf("receive %d sign in total at: %f\n", signs_in_one_frame.second.size(), estimator.Headers[k].stamp.toSec());
                        for (auto sign_in_one_frame : signs_in_one_frame.second)
                        {
                            if (sign_in_one_frame.signclass[0] == 'c')
                            {
                                keyframe = true;
                                // threads.emplace_back(threadFunction, std::ref(sign_in_one_frame), std::ref(intrinsic), std::ref(distortion), std::ref(k), std::ref(used_sign_number), std::ref(used_sign));
                                if (process_Ellipse(sign_in_one_frame, intrinsic, distortion, k))
                                {
                                    printf("cir sign %s detected!\n", sign_in_one_frame.signclass.c_str());
                                    used_sign_number++;
                                    used_sign.push_back(sign_in_one_frame);
                                }
                            }
                            else if (sign_in_one_frame.signclass[0] == 't')
                            {
                                keyframe = true;
                                // threads.emplace_back(threadFunction, std::ref(sign_in_one_frame), std::ref(intrinsic), std::ref(distortion), std::ref(k), std::ref(used_sign_number), std::ref(used_sign));
                                if (process_Tri(sign_in_one_frame, intrinsic, distortion, k))
                                {
                                    printf("tri sign %s detected!\n", sign_in_one_frame.signclass.c_str());
                                    used_sign_number++;
                                    used_sign.push_back(sign_in_one_frame);
                                }
                            }
                            else if (sign_in_one_frame.signclass[0] == 'r')
                            {
                                keyframe = true;
                                // threads.emplace_back(threadFunction, std::ref(sign_in_one_frame), std::ref(intrinsic), std::ref(distortion), std::ref(k), std::ref(used_sign_number), std::ref(used_sign));
                                if (process_Rect(sign_in_one_frame, intrinsic, distortion, k))
                                {
                                    printf("rect sign %s detected!\n", sign_in_one_frame.signclass.c_str());
                                    used_sign_number++;
                                    used_sign.push_back(sign_in_one_frame);
                                }
                            }
                        }
                        if (used_sign_number > 1 && !used_sign.empty())
                        {
                            estimator.multi_sign = true;
                            solveSignPnP(used_sign, intrinsic, distortion, k);
                            used_sign.clear();
                        }
                    }
                }
            }

            /******************************************************/
            // 20240324_xjl
            while ((!ellipse_buf.empty()) && (img_msg->header.stamp.toSec() > (ellipse_buf.front()->header.stamp.toSec() + 0.5)))
            {
                ellipse_buf.pop();
            }
            while (!tri_buf.empty() && (img_msg->header.stamp.toSec() > tri_buf.front()->header.stamp.toSec() + 0.5))
            {
                tri_buf.pop();
            }
            while (!rect_buf.empty() && (img_msg->header.stamp.toSec() > rect_buf.front()->header.stamp.toSec() + 0.5))
            {
                rect_buf.pop();
            }

            /******************************************************/
            // 20240224 xjl  aruco
            // if(estimator.timeup(ellipse_buf.front()->header.stamp.toSec()))
            Vector3d c_pos_aruco;                                                                    // aruco路标在相机坐标系下的坐标
            Vector3d n_pos_aruco;                                                                    // aruco路标在世界系下的坐标
            Quaterniond c_q_aruco;                                                                   // 路标在相机坐标系下的姿态
            Quaterniond n_q_aruco;                                                                   // 路标在世界系下的姿态
            double aruco_time;                                                                       // aruco路标帧的时间戳
            if (processArUco(aruco_buf, c_pos_aruco, n_pos_aruco, c_q_aruco, n_q_aruco, aruco_time)) // 得到时间戳、相机系下路标坐标（右前上）、世界系下坐标（右前上）和单应矩阵
            {
                // ROS_INFO("meet aruco!!!");
                std::cout << n_pos_aruco << std::endl;
                std::cout << n_q_aruco.toRotationMatrix() << std::endl;
                std::cout << n_pos_aruco << std::endl;
                std::cout << n_q_aruco.toRotationMatrix() << std::endl;
            }
            /******************************************************/

            m_buf.lock();
            estimator.processImage(image, img_msg->header, gnss_buf, keyframe, used_sign); // 主要处理函数,路标残差,gnss残差加入优化
            m_buf.unlock();
            /******************************************************/
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t); // 打印信息
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world"; // 地图系下
            // std::cout<<Utility::R2ypr(estimator.Rs[WINDOW_SIZE]).x()<<std::endl;;//输出航向角
            pubOdometry(estimator, header);   // 发布里程计话题
            pubKeyPoses(estimator, header);   // 发布关键位姿
            pubCameraPose(estimator, header); // 发布相机位姿
            pubPointCloud(estimator, header); // 发布地图点云
                                              // 20231006_xjl

            pub_Ellipse_Marker(estimator, header);   // 发布标志
            pub_Triangle_Marker(estimator, header);  // 发布标志
            pub_Rectangle_Marker(estimator, header); // 发布标志

            // pubSignPoses(estimator, header); // 发布地图点云

            pubTF(estimator, header); // 发布TF话题
            pubKeyframe(estimator);   // 发布关键帧话题
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            if (estimator.addsign)
            {
                pubSignRelocalization(estimator);
            }
            // ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{

    cv_bridge::CvImageConstPtr ptr;
    cv::Mat img;
    pair<cv::Mat, double> img_;
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
    }
    else if (img_msg->encoding == "bgr8")
    {
        img = cv::Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
        memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
    }
    else if (img_msg->encoding == "rgb8")
    {
        img = cv::Mat(static_cast<int>(img_msg->height), static_cast<int>(img_msg->width), CV_8UC3);
        memcpy(img.data, img_msg->data.data(), img_msg->height * img_msg->width * 3);
        // ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);//灰度图
        // ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); // 灰度图
        img = ptr->image;
    }
    img_.first = img;
    img_.second = img_msg->header.stamp.toSec();
    estimator.img_buf.push_back(img_);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 500, imu_callback, ros::TransportHints().tcpNoDelay());
    // ros::Subscriber sub_img = n.subscribe("/left_camera/image", 20, img_callback);
    ros::Subscriber sub_img = n.subscribe("/camera/color/image_raw", 30, img_callback); // realsensed435
    // ros::Subscriber sub_img = n.subscribe("/cam0/image_raw", 30, img_callback);

    // ros::Subscriber sub_gnss = n.subscribe("/control", 20, gnss_callback);
    // ros::Subscriber sub_gnss = n.subscribe("/gps/fix", 100, gnss_callback);
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // ros::Subscriber sub_ellipse = n.subscribe("/feature_tracker/ellipse", 30, ellipse_callback);
    // ros::Subscriber sub_tri = n.subscribe("/feature_tracker/tri", 30, tri_callback);
    // ros::Subscriber sub_rect = n.subscribe("/feature_tracker/rect", 30, rect_callback);

    ros::Subscriber sub_sign = n.subscribe("/feature_tracker/sign", 50, sign_callback);

    // ros::Subscriber sub_aruco = n.subscribe("/aruco_single/pose", 20, aruco_callback);
    // ros::Subscriber sub_map = n.subscribe("/map_pose", 20, map_callback); // 点云地图定位结果，用作真值，经纬高坐标系下

    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 200, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 200, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}