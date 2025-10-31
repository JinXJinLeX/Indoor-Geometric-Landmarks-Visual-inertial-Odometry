#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "utility/Random.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

// #include <lanelet2_core/LaneletMap.h>
// #include <lanelet2_core/primitives/BasicRegulatoryElements.h>
// #include <lanelet2_core/primitives/Lanelet.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

// 20221022xjl
#include <nav_msgs/Odometry.h>
#include "factor/gnss_factor.h"
#include "factor/sign_factor.h"
#include "factor/new_sign_factor.h"
#include "common/types.h"
#include "map_manager.h"

class Estimator
{
public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, std::queue<GNSS> &gnss_buf, bool is_sign, std::vector<SIGN> &used_sign);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // 20221022 xjl
    //  bool gvinsInitialization(GNSS gnss_,std::vector<IMU> ins_window);
    // 20221104 xjl

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    // 20221204 xjl
    void addGNSS(std::queue<GNSS> &gnss_buf);
    void addSIGN(std::queue<SIGN> &sign_buf);
    void optimization_gnss();
    void optimization_sign();
    void double2vector_();
    // 20221213 xjl
    bool sign2local(double time, int &index, Eigen::Vector3d c_pos_ellipse, Eigen::Vector3d &n_pos_ellipse);
    bool timeup(double ellipse_detec_time);
    bool isTheSameTimeNode(double time0, double time1, double interval);
    void computeEllipseLineIntersection(Vector3d g_, cv::Point2d p, Vector3d ep, double a, double b, double ori, std::vector<cv::Point2d> &pc, std::vector<Vector3d> &ps, std::vector<Vector3d> &pw, cv::Mat intrinsic, int index, double dis, Vector3d &N_);
    double computeTri(vector<cv::Point2d> &p, Vector3d &ep, Vector3d &np, Matrix3d intrinsic, double trued, int index, Vector3d g);
    // void computeRT(std::vector<Vector2d> last_pts, std::vector<Vector2d> pts, Vector3d &thita, Vector3d &p);
    bool GetSignTheta(const vector<Vector2d> pre_uv, const vector<Vector2d> cur_uv, const vector<cv::Point2d> &Hostfeat, const vector<cv::Point2d> &Targfeat, const Matrix4d &Trw, const Matrix4d &Tcw, const int &idx, Vector3d &theta, Matrix3d K);
    bool GetRANSACIdx(const int &MaxIterations, const int &SelectNum, const int &number, const bool &TEXT, vector<vector<size_t>> &IdxOut);
    double SolveTheta(const vector<size_t> &Idx, const Matrix4d &Tcr, const vector<cv::Point2d> &Targfeat, const vector<Vector3d> &HostRay, const Matrix3d &K, Vector3d &theta);
    bool choose_right(vector<cv::Point2d> input_pts1, vector<cv::Point2d> input_pts2);
    void optimization_new_sign();
    void setSignFrame(double stamp);
    std::vector<nav_msgs::Odometry> loadOdometryFromTumFile(const std::string &file_path);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    LocalMapManager map_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    // 20230922_xjl
    // vector<Vector3d> sign_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    // relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;

    // 20221022_xjl
    //  std::deque<IMU> ins_window_;
    Eigen::Vector3d ini_P;
    Eigen::Quaterniond ini_Q;
    Eigen::Matrix3d ini_R;
    // 20221202_xjl
    // GNSS gnss[WINDOW_SIZE];
    bool gnssisok[WINDOW_SIZE];
    SIGN sign[WINDOW_SIZE];
    bool signisok[WINDOW_SIZE];
    double sign_pose[WINDOW_SIZE][SIZE_POSE];
    double time[WINDOW_SIZE];
    bool first_gnss_op;
    bool first_sign_op;
    std::deque<GNSS> gnsslist_;
    std::deque<SIGN> signlist_;
    Vector3d antlever_;
    double MINIMUM_TIME_INTERVAL = 0.0045;
    std::deque<Vector3d> n_ellipse;

    std::deque<pair<cv::Mat, double>> img_buf;
    // LocalMapManager map_manager;
    // std::deque<SIGN> signlist;
    std::deque<POLE> polelist;
    std::vector<SIGN> mapforsign;
    double para_sign_Pose[NUM_OF_F][SIZE_SIGN_POSE];

    typedef struct R_T
    {
        Matrix3d R;
        Vector3d T;
        double time;
    } RT;
    std::deque<RT> RT_from_signs;
    bool sign_in_window[WINDOW_SIZE];
    bool multi_sign;
    // sign
    double sign_frame_stamp;
    Vector3d sign_relative_t;
    Quaterniond sign_relative_q;
    double sign_relative_yaw;
    int sign_frame_index;
    Matrix3d drift_sign_correct_r;
    Vector3d drift_sign_correct_t;
    bool addsign;
    std::vector<nav_msgs::Odometry> groundtruth;
};
