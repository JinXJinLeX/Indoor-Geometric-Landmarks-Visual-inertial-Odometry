#include "estimator.h"

Estimator::Estimator() : f_manager{Rs}, map_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

// Estimator::Estimator() : f_manager{Rs}
// {
//     ROS_INFO("init begins");
//     clearState();
// }

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    map_manager.setRic(ric);
    ProjectionFactor::sqrt_info = 2 * FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = 2 * FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    NEWSIGNFactor::sqrt_info = FOCAL_LENGTH / 0.25 * Matrix<double, 10, 10>::Identity();
    NEWRSIGNFactor::sqrt_rinfo = FOCAL_LENGTH / 0.25 * Matrix<double, 18, 18>::Identity();
    NEWTSIGNFactor::sqrt_tinfo = FOCAL_LENGTH / 0.25 * Matrix<double, 10, 10>::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();
    map_manager.clearsignState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

bool Estimator::GetRANSACIdx(const int &MaxIterations, const int &SelectNum, const int &number, const bool &TEXT, vector<vector<size_t>> &IdxOut)
{
    if (TEXT)
    {
        if (number < 3)
            return false;
    }

    IdxOut = vector<vector<size_t>>(MaxIterations, vector<size_t>(SelectNum, 0));

    DUtils::Random::SeedRandOnce(0);
    for (int it = 0; it < MaxIterations; it++)
    {
        vector<size_t> vAvailableIndices = DUtils::Random::InitialVec(number);
        for (size_t j = 0; j < SelectNum; j++)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randi];

            IdxOut[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    return true;
}

double Estimator::SolveTheta(const vector<size_t> &Idx, const Matrix4d &Tcr, const vector<cv::Point2d> &Targfeat, const vector<Vector3d> &HostRay, const Matrix3d &K, Vector3d &theta)
{
    const float th = 5.991;
    assert(Idx.size() == 3);

    Vector3d m1 = HostRay[Idx[0]];
    Vector3d m2 = HostRay[Idx[1]];
    Vector3d m3 = HostRay[Idx[2]];

    // 1. get solution
    // 0 0 D    theta1   rho1'
    // 0 1 c    theta2   rho2'
    // 1 A B    theta3   rho3'
    double A = m3[1] / m3[0];
    double B = 1.0 / m3[0];
    double C = (1 - m2[0] * B) / (m2[1] - m2[0] * A);
    double D = (A * C - B) * m1[0] - C * m1[1] + 1;
    double rho3pie = m3[2] / m3[0];
    double rho2pie = (m2[2] - rho3pie * m2[0]) / (m2[1] - m2[0] * A);
    double rho1pie = m1[2] - rho3pie * m1[0] - rho2pie * (m1[1] - A * m1[0]);
    theta(2, 0) = rho1pie / D;
    theta(1, 0) = rho2pie - C * theta(2, 0);
    theta(0, 0) = rho3pie - A * theta(1, 0) - B * theta(2, 0);

    double score = 0;
    int numOK = 0;
    for (size_t ipts = 0; ipts < HostRay.size(); ipts++)
    {
        Vector3d m(HostRay[ipts](0), HostRay[ipts](1), 1.0);
        double rhoPred = m.transpose() * theta;
        Vector3d P3d = Tcr.block<3, 3>(0, 0) * m * (1.0 / rhoPred) + Tcr.block<3, 1>(0, 3);
        double u = P3d(0) / P3d(2) * K(0, 0) + K(0, 2);
        double v = P3d(1) / P3d(2) * K(1, 1) + K(1, 2);
        double erroru = Targfeat[ipts].x - u;
        double errorv = Targfeat[ipts].y - v;
        double chiSquare2 = erroru * erroru + errorv * errorv;

        if (chiSquare2 > th)
            continue;
        else
            score += th - chiSquare2;

        numOK++;
    }

    theta = -theta;

    return score;
}

bool Estimator::GetSignTheta(const vector<Vector2d> pre_uv, const vector<Vector2d> cur_uv, const vector<cv::Point2d> &Hostfeat, const vector<cv::Point2d> &Targfeat, const Matrix4d &Trw, const Matrix4d &Tcw, const int &idx, Vector3d &theta, Matrix3d K)
{
    Matrix4d Tcr = Tcw * Trw.inverse();

    // 1. param initial
    cv::Mat T1 = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat T2 = cv::Mat::eye(4, 4, CV_64F);
    T2 = (cv::Mat_<double>(4, 4) << Tcr(0, 0), Tcr(0, 1), Tcr(0, 2), Tcr(0, 3),
          Tcr(1, 0), Tcr(1, 1), Tcr(1, 2), Tcr(1, 3),
          Tcr(2, 0), Tcr(2, 1), Tcr(2, 2), Tcr(2, 3),
          0, 0, 0, 1);

    vector<cv::Point2d> ptCam1, ptCam2;
    cv::Mat NewPts4d;
    for (size_t i3DIni = 0; i3DIni < Hostfeat.size(); i3DIni++)
    {
        ptCam1.push_back(cv::Point2d(pre_uv[i3DIni].x(), pre_uv[i3DIni].y()));
        ptCam2.push_back(cv::Point2d(cur_uv[i3DIni].x(), cur_uv[i3DIni].y()));
    }

    // 2. triangulation
    cv::triangulatePoints(T1.rowRange(0, 3).colRange(0, 4), T2.rowRange(0, 3).colRange(0, 4), ptCam1, ptCam2, NewPts4d);

    // 3. get (xyz) & (rho)
    vector<Vector3d> NewPts3dRaw;
    vector<double> vRho;
    for (size_t i3DProc = 0; i3DProc < NewPts4d.cols; i3DProc++)
    {
        cv::Mat p4d = NewPts4d.col(i3DProc);
        p4d /= p4d.at<float>(3, 0);
        Vector3d p3d((double)p4d.at<float>(0, 0), (double)p4d.at<float>(1, 0), (double)p4d.at<float>(2, 0));
        NewPts3dRaw.push_back(p3d);
        vRho.push_back(1.0 / p3d(2, 0));
    }

    // 5. use rho calculate theta
    vector<Vector3d> vHostRayRho;
    // Tool.GetRayRho(Hostfeat, vRho, cfCurrentFrame.mK, vHostRayRho);

    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    for (size_t i = 0; i < Hostfeat.size(); i++)
    {
        Vector3d ray((Hostfeat[i].x - cx) / fx, (Hostfeat[i].y - cy) / fy, vRho[i]);
        vHostRayRho.push_back(ray);
    }
    // 5.1. get RANSAC Idx
    int iMaxIterations = 10, iSelectnum = 3;
    vector<vector<size_t>> RansacIdx;
    bool NUMOK = GetRANSACIdx(iMaxIterations, iSelectnum, Hostfeat.size(), true, RansacIdx);
    if (!NUMOK)
        return false;

    // 5.2 calculate theta using rho
    double BestScore = -1.0;
    Vector3d BestTheta(-1.0, -1.0, -1.0);
    for (size_t i0 = 0; i0 < RansacIdx.size(); i0++)
    {

        vector<size_t> Idx = RansacIdx[i0];

        Vector3d thetaObj;
        double score = SolveTheta(Idx, Tcr, Targfeat, vHostRayRho, K, thetaObj);

        if (BestScore < score)
        {
            BestTheta = thetaObj;
            BestScore = score;
            // cout << BestScore << endl;
        }
    }
    if (BestScore != -1)
    {
        theta = BestTheta;
    }
    else
    {
        return false;
    }

    return true;
}
std::vector<nav_msgs::Odometry> Estimator::loadOdometryFromTumFile(const std::string &file_path) {
    std::vector<nav_msgs::Odometry> odometry_msgs;
    std::ifstream file(file_path);
    if (!file.is_open()) {
      ROS_ERROR("Failed to open TUM file: %s", file_path.c_str());
      return odometry_msgs;
    }
  
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;  // 跳过空行和注释行
      }
  
      nav_msgs::Odometry odometry_msg;
      std::istringstream ss(line);
  
      double timestamp;
      ss >> timestamp >> odometry_msg.pose.pose.position.x >>
          odometry_msg.pose.pose.position.y >>
          odometry_msg.pose.pose.position.z >>
          odometry_msg.pose.pose.orientation.x >>
          odometry_msg.pose.pose.orientation.y >>
          odometry_msg.pose.pose.orientation.z >>
          odometry_msg.pose.pose.orientation.w;
  
      // 设置时间戳和帧 ID
      odometry_msg.header.stamp = ros::Time(timestamp);
      odometry_msg.header.frame_id = "world";
  
      odometry_msgs.push_back(odometry_msg);
    }
  
    file.close();
    ROS_INFO("Loaded %lu odometry messages from TUM file.", odometry_msgs.size());
    return odometry_msgs;
  }

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 1. 判断是不是第一个imu消息，如果是第一个imu消息，就将当前的加速度和角速度作为初始加速度和角速度
    if (!first_imu) // first_imu 标志位，初始值为false
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    /* 2. 创建预积分对象
        pre_integrations[]是一个数组，存放了WINDOW_SIZE+1个指针，指针指向的类型是IntegrationBase
        */
    // 如果预积分为0
    if (!pre_integrations[frame_count]) //**之前没给pre_integrations[frame_count]赋值条件就为真**
    {
        // 当滑窗还没满的时候，或者在移动滑窗的时候，就需要new一个pre_integration
        //  acc_0  gyr_0 是前一时刻的加速度和角速度
        // 对其初始化
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // frame_count == 0表示窗口中还没有图像帧，所以不进行预积分
    if (frame_count != 0)
    {
        // 3. 进行预积分
        // 对第frame_count中的预积分片段进行数据填充
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // 临时预积分初始值	//这个push_back是自己写的一个函数，其中包含了雅各比的传播

        // 对dt缓存、加速度、角速度进行数据填充
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        /* 4. 更新Rs Ps Vs 三个向量组 */
        // 针对第frame_count帧img内的预积分片段进行积分
        int j = frame_count;
        // 计算上一时刻的加速度，前面乘一个 Rs  旋转到第一帧IMU的坐标系
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // a_0 = Rs[j] * ( a0 - ba[j]) - g，即去掉零偏后的加表数值作用于j时刻的位姿上的加速度，再减去重力影响
        // 根据上一个时刻的角速度和当前时刻的角速度求出平均角速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; // w_0 = 0.5(w0 + w) - bg[j]，即原角速度和陀螺仪数值的中值减去零偏
        // 计算当前时刻陀螺仪的旋转矩阵，是在上一时刻的旋转矩阵的基础上和当前时刻的旋转增量相乘得到的
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); // 当前估计位姿
        // 当前时刻的加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g; // 去掉零偏后的加表数值作用于当前位姿上的加速度，再减去重力影响
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);                  // 中值积分
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;                   // 位置更新
        Vs[j] += dt * un_acc;                                           // 速度更新
    }
    // 更新a_0和w_0
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, std::queue<GNSS> &gnss_buf, bool is_sign, std::vector<SIGN> &used_sign)
{

    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // printf("\033[1;32m ???%f \n\033[0m", ellipse_time);
    // printf("\033[1;31m %f \n\033[0m", header.stamp.toSec());
    if (is_sign || f_manager.addFeatureCheckParallax(frame_count, image, td)) // 判断平均视差是否够大，够大成为新的关键帧，就要边缘化最前的老帧    标志和当前帧同帧，算关键帧
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    ROS_DEBUG("number of sign: %d", map_manager.getSignCount());

    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration; // tmp_pre_integration 是在processIMU() 时进行的  IntegrationBase *tmp_pre_integration;
    // 存储所有的imageFrame
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 2. 外参标定
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 1. 获取两帧之间的匹配的点对
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 2. 计算相机到IMU的旋转矩阵
            // pre_integrations[frame_count-1]->delta_q 是IMU预积分得到的旋转  calib_ric是要计算相机到IMU的旋转
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                // 外参标定的结果存储在ric[0], RIC[0]
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                // 外参初始化成功
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 3. 初始化
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE) // 滑窗满了再初始化
        {
            bool result = false;
            // 外参初始化成功
            if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // 对视觉个惯性单元进行初始化
                result = initialStructure();
                // 初始化时间戳更新
                initial_timestamp = header.stamp.toSec();
                first_gnss_op = true;
                first_sign_op = true;
            }
            if (result) // 初始化成功
            {
                // 先进行一次滑动窗口，非线性优化，得到当前帧与第一帧的相对位姿
                solver_flag = NON_LINEAR; // 初始化成功之后，就转成非线性优化
                // addSIGN(sign_buf);   // 将sign_buf中合适的节点加入因子图优化
                // optimization_sign(); // 进行优化求解
                // double2vector();
                solveOdometry();
                addGNSS(gnss_buf);
                optimization_gnss();
                double2vector();
                slideWindow();
                f_manager.removeFailures();
                // map_manager.removesignOutlier();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                // 20221202xjl
                // ROS_INFO("before addGNSS!!! ");
                // ROS_INFO("addGNSS!!! ");
                // addGNSS(gnss_buf);
                // optimization_gnss();
                // addSIGN(sign_buf);   // 将sign_buf中合适的节点加入因子图优化
                // optimization_sign(); // 进行优化求解
                // double2vector();
                // solver_flag = NON_LINEAR; //初始化成功之后，就转成非线性优化
            }
            else
                slideWindow();
        }
        else
        {
            frame_count++;
        }
    }
    // 4. 非线性优化
    else
    {
        TicToc t_solve;

        solveOdometry();

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // if (failureDetection())
        // {
        //     ROS_WARN("failure detection!");
        //     failure_occur = 1;
        //     clearState();
        //     setParameter();
        //     ROS_WARN("system reboot!");
        //     return;
        // }
        addGNSS(gnss_buf);
        optimization_gnss();
        // double2vector();
        // addSIGN(sign_buf);   // 将sign_buf中合适的节点加入因子图优化
        // optimization_sign(); // 进行优化求解
        // double2vector();
        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        // map_manager.removesignOutlier();

        // addGNSS(gnss_buf);
        // optimization_gnss();
        double2vector();
        // addSIGN(sign_buf); // 将sign_buf中合适的节点加入因子图优化
        // optimization_sign(); // 进行优化求解
        // double2vector();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);
        // 20230922_xjl
        // sign_poses.clear();
        // for (int i = 0; i <= WINDOW_SIZE; i++)
        //     sign_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 遍历除了第一帧之外的所有图像帧
        // 图像帧里包含了时间戳、image     和预积分结果做初始化
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            // 时间
            double dt = frame_it->second.pre_integration->sum_dt;
            // 计算每一帧图像对应的加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 图像的加速度累加
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        // 计算平均加速度
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        // 遍历除了第一帧之外的所有图像帧
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 方差：加速度 减去 平均加速度 的      差值的   平方的    累加
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 标准差
        /**
         * 方差公式：s^2 = [(x1 - x)^2 + (x2 - x)^2 + ... + (xn - x)^2]/n
         * 标准差： s = sqrt(s^2)
         */
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        // ROS_WARN("IMU variation %f!", var);
        // 通过加速度标准差判断IMU是否由充分运动，标准差必须大于等于0.25
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            // return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    // 将f_manager.feature中的  feature存储到 sfm_f 中
    for (auto &it_per_id : f_manager.feature) // list<FeaturePerId> feature;
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id; // 就是特征点的id
        // 遍历每一个能观察到该feature的frame
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; // 后面要++，前面为啥要-1呢？ imu_j就是观测到点的帧的序号
            Vector3d pts_j = it_per_frame.point;
            // 每个特征点能被那些帧观测到，以及特征点在这些帧中的坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // 通过求解本质矩阵来求解位姿
    /**
     * l 表示滑动窗口中第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧
     */
    if (!relativePose(relative_R, relative_T, l)) // 计算的结果给relative_R, relative_T
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            std::cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::visualInitialAlign() // imu视觉对齐
{
    TicToc t_g;
    VectorXd x;
    // solve scale 1. 视觉惯性联合初始化， 计算陀螺仪的偏置，尺度，重力加速度和 速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state 2. 获取滑动窗口内所有帧相对于第l帧的位姿信息，并设置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 3. 获取特征点深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 三角化计算地图点的深度，Ps 中存放的是 各个帧相对于参考帧之间的平移，RIC[0] 为 相机-IMU 之间的旋转
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // 4. 这里陀螺仪的偏差Bags 改变了，需要遍历滑窗中的帧，重新进行预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 5. 计算各帧相对于b0的位置信息，前面计算的都是相对于第l帧的位姿，在这里转换到b0帧坐标系下
    /*
     * 前面的初始化中，计算出来的是相对滑动窗中第l帧的位姿，在这里转换到b0帧的坐标系下
     * s*p_bk^​b0​​=s*p_bk^​cl​​−s*p_b0^​cl​​=(s*p_ck^​cl​​−R_bk​^cl​​*p_c^b​)−(s*p_c0^​cl​​−R_b0​^cl​​*p_c^b​)
     * TIC[0]是相机到IMU的旋转量
     * Rs 是IMU第k帧到滑动窗口中图像第l帧的旋转
     * Ps 是滑动窗口中第k帧到第l帧的平移
     * 如果 launch文件中配置的 无外参，那么TIC 都是 0
     */
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            // 存储速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 更新每个地图点被观测到的帧数(used_num) 和预测的深度 (estimated_depth)
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    /**
     * refine 之后就获得了C_0坐标系下的中立g^{c_0},此时通过将g^{c_0}旋转值z轴方向
     * 这样就可以i计算相机系到世界系的旋转矩阵 q_{c_0}^w, 这里求得的是rot_diff， 这样就可以将所有的变量调整到世界系中
     */
    Matrix3d R0 = Utility::g2R(g);
    // 这里调整为东北天坐标系（右前上）但是我们的yaw使用北偏西为正
    double yaw = Utility::R2ypr(ini_R).x();
    // double yaw = -Utility::R2ypr(R0 * Rs[0]).x() + Utility::R2ypr(ini_R).x();
    // cout << Utility::R2ypr(R0 * Rs[0]).x() << endl;
    // cout << Utility::R2ypr(ini_R).x() << endl;
    // cout << yaw << endl;
    // R0 = Utility::ypr2R(Eigen::Vector3d{yaw, 0, 0}) * R0;
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    // 20221208 xjl 初始化
    // Matrix3d rot_diff = ini_R;
    Matrix3d rot_diff = R0;

    // Vector3d V;
    // V = rot_diff.eulerAngles(0, 1, 2);
    // cout << "RotationMatrix result is:" << endl;
    // cout << R0 << endl;
    // cout << Utility::R2ypr(R0).x() << endl;

    // cout << "RotationMatrix2euler result is:" << endl;
    // cout << "x = " << V[2] << endl;
    // cout << "y = " << V[1] << endl;
    // cout << "z = " << V[0] << endl;
    // 所有的变量从参考坐标系c_0 旋转到世界坐标系 w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i] + ini_P;
        // Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        // Rs[i] = rot_diff;
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * 这里的第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧
 * 会作为参考帧到下面的全局sfm使用，得到的 R t 为当前帧到第l帧的坐标变换 R t
 * 该函数判断滑动窗口中第一帧到最后一帧，对应特征点的平均视差大于30，且内点数大于12的帧，此时可进行初始化，同时返回当前帧到第l帧的坐标变化 R t
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 寻找第 i 帧 到最后一帧对应的特征点，存放在corres中，从第一帧开始找，到最先符合条件的一个帧
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 第j个对应点在第 i 帧和最后一帧的（x,y）
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            // 平均视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 判断是否满足初始化条件：视差> 30， 内点数>12 solveRelativeRT 返回true
            //  solveRelativePoseRtT() 通过基础矩阵计算当前帧与第l帧的 R 和 T ，并判断内点数是否足够
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的relative_R 和relativeT
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        // optimization_new_sign();
        optimization();
    }
}

// 20230703_xjl
void Estimator::addSIGN(std::queue<SIGN> &sign_buf)
{
    if (frame_count < WINDOW_SIZE)
        return;
    SIGN sign_;
    double time_;
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        time[i] = Headers[i].stamp.toSec();
    }
    while (!sign_buf.empty())
    {
        sign_ = sign_buf.front();
        time_ = sign_.time;
        if (time_ <= time[0])
        {
            sign_buf.pop();
        }
        else
            break;
    }
    while ((!signlist_.empty()) && (time[0] > signlist_.front().time))
    {
        signlist_.pop_front();
        // ROS_WARN("throw sign!!");
    }
    if (marginalization_flag == MARGIN_OLD)
    {
        while (!sign_buf.empty())
        {
            sign_ = sign_buf.front();
            time_ = sign_.time;
            if (time[WINDOW_SIZE - 1] >= time_ && time_ >= time[0])
            {
                // ROS_WARN("join sign!!");
                sign_buf.pop();
                signlist_.push_back(sign_);
                continue;
            }
            else
                break;
        }
    }
    else if (marginalization_flag == MARGIN_SECOND_NEW)
    {
        while (!sign_buf.empty())
        {
            sign_ = sign_buf.front();
            time_ = sign_.time;
            if (time[WINDOW_SIZE - 1] >= time_ && time_ >= time[0])
            {
                sign_buf.pop();
                signlist_.push_back(sign_);
                continue;
            }
            else
                break;
        }
    }

    // if (sign_buf.empty())
    // {
    //     return;
    // }
    // if (frame_count < WINDOW_SIZE)
    //     return;
    // SIGN sign_;
    // double time_;
    // // map<double, ImageFrame>::iterator frame_it;
    // if (first_sign_op)
    // {
    //     for (int i = 0; i < WINDOW_SIZE; i++)
    //     {
    //         time[i] = Headers[i].stamp.toSec();
    //     }
    //     for (int k = 0; k < WINDOW_SIZE; k++)
    //     {
    //         while (!sign_buf.empty())
    //         {
    //             sign_ = sign_buf.front();
    //             sign_buf.pop();
    //             time_ = sign_.time;
    //             if (time[k] - time_ < 0.01 && time[k] > time_)
    //             {
    //                 sign[k] = sign_;
    //                 break;
    //             }
    //         }
    //     }
    //     first_sign_op = 0;
    // }
    // else
    // {
    //     for (int i = 0; i < WINDOW_SIZE; i++)
    //     {
    //         time[i] = Headers[i].stamp.toSec();
    //     }
    //     if (time[0] - sign[0].time > 0.01)
    //     {
    //         for (int i = 0; i < WINDOW_SIZE - 1; i++)
    //         {
    //             sign[i] = sign[i + 1];
    //         }
    //     }
    //     while (!sign_buf.empty())
    //     {
    //         sign_ = sign_buf.front();
    //         sign_buf.pop();
    //         time_ = sign_.time;
    //         if (time[WINDOW_SIZE - 1] - time_ < 0.01 && time[WINDOW_SIZE - 1] > time_)
    //         {
    //             sign[WINDOW_SIZE - 1] = sign_;
    //             break;
    //         }
    //     }
    // }
}

// 20221202xjl
void Estimator::addGNSS(std::queue<GNSS> &gnss_buf) // 将gnss_buf中的GNSS帧存入vector
{
    if (gnss_buf.empty())
    {
        printf("return!!!!");
        return;
    }
    if (frame_count < WINDOW_SIZE)
        return;
    GNSS gnss_;
    double time_;
    // map<double, ImageFrame>::iterator frame_it;
    // 先拿出时间戳
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        time[i] = Headers[i].stamp.toSec();
    }
    // if (first_gnss_op)
    // {
    //     first_gnss_op = false;
    while (!gnss_buf.empty())
    {
        gnss_ = gnss_buf.front();
        time_ = gnss_.time;
        if (time_ <= time[0])
        {
            gnss_buf.pop();
        }
        else
            break;
    }
    while ((!gnsslist_.empty()) && (time[0] > gnsslist_.front().time))
    {
        gnsslist_.pop_front();
        // ROS_WARN("throw gnss!!");
    }
    if (marginalization_flag == MARGIN_OLD)
    {
        while (!gnss_buf.empty())
        {
            gnss_ = gnss_buf.front();
            time_ = gnss_.time;
            if (time[WINDOW_SIZE - 1] >= time_ && time_ >= time[0])
            {
                // ROS_WARN("join gnss111!!");
                gnss_buf.pop();
                gnsslist_.push_back(gnss_);
                continue;
            }
            else
                break;
        }
    }
    else if (marginalization_flag == MARGIN_SECOND_NEW)
    {
        // while ((!gnsslist_.empty()) && (time[0] > gnsslist_.front().time))
        // {
        //     // ROS_WARN("pop gnss111!!");
        //     gnsslist_.pop_front();
        // }
        while (!gnss_buf.empty())
        {
            gnss_ = gnss_buf.front();
            time_ = gnss_.time;
            if (time[WINDOW_SIZE - 1] >= time_ && time_ >= time[0])
            {
                gnss_buf.pop();
                gnsslist_.push_back(gnss_);
                continue;
            }
            // else if (time_ > time[WINDOW_SIZE])
            // {
            //     break;
            // }
            else
                break;
        }
    }
}

// // 新优化 xjl
void Estimator::optimization_new_sign()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function1;
    loss_function1 = new ceres::CauchyLoss(1.0);
    // loss_function1 = new ceres::TrivialLoss;
    vector2double(); // 将系统状态向量转为double类型数组以适应ceres
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // 优化P和Q
        // problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);               // 优化速度和零偏V ba bg
        // problem.SetParameterBlockConstant(para_SpeedBias[i]);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); // 优化外参ric,tic
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);                                                   // 预积分
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]); // IMU因子残差块，可以看到有前后帧的联系
    }
    // add sign factor
    int s_m_cnt = 0;
    bool addsign = false;
    for (auto &it_per_id : map_manager.sign) // 遍历滑窗内的标志
    {
        if (it_per_id.used_num != 0 && !it_per_id.sign_per_frame.empty())
        {
            int frame_cnt = 0;
            for (auto &it_per_frame : it_per_id.sign_per_frame) // 该标志在滑窗内每一帧（可能不是）的信息
            {
                SIGN reds_sign;
                reds_sign.signclass = it_per_id.classify;
                reds_sign.N = it_per_id.N_;
                reds_sign.C = it_per_id.C_;
                reds_sign.cvPoints = it_per_frame.pts;
                reds_sign.time = it_per_frame.time;
                reds_sign.scale = 0.3;
                for (int i = 0; i < WINDOW_SIZE; i++)
                {
                    if (isTheSameTimeNode(reds_sign.time, Headers[i].stamp.toSec(), MINIMUM_TIME_INTERVAL))
                    {
                        problem.SetParameterBlockConstant(para_SpeedBias[i]);
                        frame_cnt = i;
                        if (ESTIMATE_SIGN && frame_cnt != 0)
                        {
                            if (reds_sign.signclass[0] == 'c')
                            {
                                addsign = true;
                                printf("add circle residual at %f, sign id is: %d,total frame:%d \n", reds_sign.time, it_per_id.sign_id, it_per_id.sign_per_frame.size());
                                // cout << "ith frame" << frame_cnt << endl;
                                // NEWSIGNFactor *newsign_factor = new NEWSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(newsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNFactor, 18, 7, 7>(new SIGNFactor(reds_sign)), loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // printf("%f", reds_sign.time);
                                // printf("%f", Headers[frame_cnt].stamp.toSec());
                                // double **para = new double *[3];
                                // para[0] = para_Pose[frame_cnt];
                                // para[1] = para_Ex_Pose[0];
                                // para[2] = para_sign_Pose[id];
                                // newsign_factor->check(para);
                            }
                            else if (reds_sign.signclass[0] == 'r')
                            {
                                addsign = true;
                                printf("add rectangle residual at %f, sign id is: %d \n", reds_sign.time, it_per_id.sign_id);
                                // NEWRSIGNFactor *rsign_factor = new NEWRSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(rsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNRFactor, 18, 7, 7>(new SIGNRFactor(reds_sign)), loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                            }
                            else if (reds_sign.signclass[0] == 't')
                            {
                                addsign = true;
                                printf("add triangle residual at %f, sign id is: %d \n", reds_sign.time, it_per_id.sign_id);
                                // NEWTSIGNFactor *tsign_factor = new NEWTSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(tsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNTFactor, 10, 7, 7>(new SIGNTFactor(reds_sign)), loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                            }
                        }
                    }
                }
            }
            // ROS_INFO("sign measurement count: %d", frame_cnt);
        }
        s_m_cnt++;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR; // 线性方程求解的方法
        options.minimizer_progress_to_stdout = true;     // 输出优化的过程
        // 优化信息结果
        ceres::Solver::Summary summary;
        // 开始优化
        ceres::Solve(options, &problem, &summary);
        // 输出优化结果简报
        cout << summary.BriefReport() << endl;
        double2vector_();
    }
}

void Estimator::optimization_sign()
{
    // for (int i = 0; i < WINDOW_SIZE; i++)
    // {
    //     cout<<"before opitimization"<<endl;
    //     cout << para_Pose[i][0] << endl;//xyz
    //     cout << para_Pose[i][1] << endl;
    //     cout << para_Pose[i][2] << endl;
    //     cout<<"gnss"<<endl;
    //     cout << gnss[i].blh << endl;//enu
    // }
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0);
    vector2double(); // 转为double类型
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE); // 加入外参状态
    }
    // for (int i = 0; i < sign_poses.size(); i++)
    // {
    //     problem.AddParameterBlock(sign_poses[i], SIZE_SIGN_POSE);//加入路标状态
    // }
    if (!signlist_.empty())
    {
        for (SIGN sign : signlist_)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                // if (sign.similarty > 0.8)
                // {
                if (isTheSameTimeNode(sign.time, Headers[i].stamp.toSec(), MINIMUM_TIME_INTERVAL))
                {
                    for (int j = 0; j < sign.cvPoints.size(); j++)
                    {
                        // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNFactor, 3, 7>(new SIGNFactor(para_Pose[i], sign)), loss_function, para_Pose[i]);
                        // sign.Points.pop_front();
                        // sign.xyz.pop_front();
                    }
                }
                // }
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // 线性方程求解的方法
    // options.minimizer_progress_to_stdout = true;  //输出优化的过程
    // 优化信息结果
    ceres::Solver::Summary summary;
    // 开始优化
    ceres::Solve(options, &problem, &summary);
    // 输出优化结果简报
    // cout << summary.BriefReport() << endl;
    //  double2vector_();
}

bool Estimator::isTheSameTimeNode(double time0, double time1, double interval)
{
    return fabs(time0 - time1) < interval;
}

void Estimator::optimization_gnss()
{
    // for (int i = 0; i < WINDOW_SIZE; i++)
    // {
    //     cout << "before opitimization" << endl;
    //     cout << para_Pose[i][0] << endl; // xyz
    //     cout << para_Pose[i][1] << endl;
    //     cout << para_Pose[i][2] << endl;
    //     // cout<<"gnss"<<endl;
    //     // cout << gnss[i].blh << endl;//enu
    // }
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0);
    vector2double();
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        // printf("camera time: %f\n", Headers[i].stamp.toSec());
    }
    if (!gnsslist_.empty())
    {
        // ROS_WARN("not empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

        for (GNSS gnss_in_list : gnsslist_)
        {
            for (size_t k = 0; k < WINDOW_SIZE; k++)
            {
                if (isTheSameTimeNode(gnss_in_list.time, Headers[k].stamp.toSec(), MINIMUM_TIME_INTERVAL))
                {
                    // printf("gnss time: %f\n", gnss_in_list.time);
                    // ROS_INFO("AddResidualGNSSBlock!!!");
                    // cout << gnss_in_list.blh << endl; // enu
                    // cout << para_Pose[k][0] << endl;  // xyz
                    // cout << para_Pose[k][1] << endl;
                    // cout << para_Pose[k][2] << endl;
                    // gnss[k] = gnss_in_list;
                    // GNSSFactor* gnssfactor = new GNSSFactor(para_Pose[k], gnss[k]);
                    // problem.AddResidualBlock(gnssfactor, loss_function, para_Pose[k]);

                    // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<GNSSFactor, 3, 7>(new GNSSFactor(para_Pose[k], gnss_in_list)), loss_function, para_Pose[k]);
                }
                // ROS_WARN("Not Solved!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // 线性方程求解的方法
    // options.minimizer_progress_to_stdout = true;  //输出优化的过程
    // 优化信息结果
    ceres::Solver::Summary summary;
    // 开始优化
    ceres::Solve(options, &problem, &summary);
    // 输出优化结果简报
    // cout << summary.BriefReport() << endl;
    //  double2vector_();
    // for (int i = 0; i < WINDOW_SIZE; i++)
    // {
    //     cout << "after opitimization" << endl;
    //     cout << para_Pose[i][0] << endl; // xyz
    //     cout << para_Pose[i][1] << endl;
    //     cout << para_Pose[i][2] << endl;
    //     // cout<<"gnss"<<endl;
    //     // cout << gnss[i].blh << endl;//enu
    // }
}

// 20230913_xjl
//  输入：在相机系下的重力g
//  标志中心像素坐标p
//  标志中心点相机系下位置ep
//  长短轴、倾斜角以及
//  得到相机平面的标志最高点、最低点、最远点、最近点(其中以靠左边的这个点，对应标志靠右的点为x轴)
void Estimator::computeEllipseLineIntersection(Vector3d g_, cv::Point2d p, Vector3d ep, double a, double b, double ori, std::vector<cv::Point2d> &pc, std::vector<Vector3d> &ps, std::vector<Vector3d> &pw, cv::Mat intrinsic, int index, double dis, Vector3d &N_)
{
    double fx = intrinsic.at<double>(0, 0);
    double fy = intrinsic.at<double>(1, 1);
    double cx = intrinsic.at<double>(0, 2);
    double cy = intrinsic.at<double>(1, 2);
    double theta = ori / 180 * M_PI;
    cv::Point2d p0, p1, p2, p3, p4;
    Vector3d P0, P1, P2, P3, P4, P5, Pa1, Pa2, Pb1, Pb2;
    double A, B, C, D, E, F;
    double x1, x2, y1, y2, x3, x4, y3, y4;
    bool vi = false;
    double k1, b1, b2, b3; // 直线斜率和截径
    double delta;
    Matrix3d intrinsic_;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            intrinsic_(i, j) = intrinsic.at<double>(i, j);
        }
    }
    pc.clear();
    ps.clear();
    pw.clear();
    p0 = p;
    p1.x = p.x + g_[0] * 5;
    p1.y = p.y + g_[1] * 5;
    k1 = (p1.y - p0.y) / (p1.x - p0.x);
    b1 = p0.y - k1 * p0.x;
    // 椭圆方程
    A = (cos(theta) / a) * (cos(theta) / a) + (sin(theta) / b) * (sin(theta) / b);
    B = 2 * (1 / (a * a) - 1 / (b * b)) * sin(theta) * cos(theta);
    C = (sin(theta) / a) * (sin(theta) / a) + (cos(theta) / b) * (cos(theta) / b);
    // A = a * a * cos(theta) * cos(theta) + b * b * sin(theta) * sin(theta);
    // B = 2 * (b * b - a * a) * sin(theta) * cos(theta);
    // C = a * a * sin(theta) * sin(theta) + b * b * cos(theta) * cos(theta);
    D = -2 * A * p0.x - B * p0.y;
    E = -B * p0.x - 2 * C * p0.y;
    F = A * p0.x * p0.x + B * p0.x * p0.y + C * p0.y * p0.y - 1;

    if (fabs(p1.x - p0.x) > 0.0001 && (pow((B * b1 + 2 * C * k1 * b1 + D + E * k1), 2) - 4 * (A + B * k1 + C * k1 * k1) * (C * b1 * b1 + E * b1 + F)) > 0)
    {
        vi = false;
        k1 = (p1.y - p0.y) / (p1.x - p0.x);
        b1 = p0.y - k1 * p0.x;
        delta = (pow((B * b1 + 2 * C * k1 * b1 + D + E * k1), 2) - 4 * (A + B * k1 + C * k1 * k1) * (C * b1 * b1 + E * b1 + F));
        // 以p0为点，g为法向量求得直线与椭圆交点
        x1 = (-(B * b1 + 2 * C * k1 * b1 + D + E * k1) + sqrt(delta)) / (2 * (A + B * k1 + C * k1 * k1));
        x2 = (-(B * b1 + 2 * C * k1 * b1 + D + E * k1) - sqrt(delta)) / (2 * (A + B * k1 + C * k1 * k1));
        y1 = k1 * x1 + b1;
        y2 = k1 * x2 + b1;
        p1 = cv::Point2d(x1, y1);
        p2 = cv::Point2d(x2, y2);
        double lamda = (A + B * k1 + C * k1 * k1);
        // 令delta为0，另列关于b的方程，求另外两条直线
        b2 = (-(2 * B * D + 2 * E * B * k1 + 4 * D * C * k1 + 4 * E * k1 * k1 * C - 4 * E * lamda) - sqrt(pow((2 * B * D + 2 * E * B * k1 + 4 * D * C * k1 + 4 * E * k1 * k1 * C - 4 * E * lamda), 2) - 4 * (B * B + 4 * C * B * k1 + 4 * C * C * k1 * k1 - 4 * C * lamda) * ((D + E * k1) * (D + E * k1) - 4 * F * lamda))) / (2 * (B * B + 4 * C * B * k1 + 4 * C * C * k1 * k1 - 4 * C * lamda));
        b3 = (-(2 * B * D + 2 * E * B * k1 + 4 * D * C * k1 + 4 * E * k1 * k1 * C - 4 * E * lamda) + sqrt(pow((2 * B * D + 2 * E * B * k1 + 4 * D * C * k1 + 4 * E * k1 * k1 * C - 4 * E * lamda), 2) - 4 * (B * B + 4 * C * B * k1 + 4 * C * C * k1 * k1 - 4 * C * lamda) * ((D + E * k1) * (D + E * k1) - 4 * F * lamda))) / (2 * (B * B + 4 * C * B * k1 + 4 * C * C * k1 * k1 - 4 * C * lamda));
        // delta = (pow((B * b2 + 2 * C * k1 * b2 + D + E * k1), 2) - 4 * (A + B * k1 + C * k1 * k1) * (C * b2 * b2 + E * b2 + F));
        // cout << delta << endl;
        x3 = -(B * b2 + 2 * C * k1 * b2 + D + E * k1) / (2 * (A + B * k1 + C * k1 * k1));
        y3 = k1 * x3 + b2;
        x4 = -(B * b3 + 2 * C * k1 * b3 + D + E * k1) / (2 * (A + B * k1 + C * k1 * k1));
        y4 = k1 * x4 + b3;
        p3 = cv::Point2d(x3, y3);
        p4 = cv::Point2d(x4, y4);
        // p5.x = p.x + a * cos(ori * M_PI / 180.0);
        // p5.y = p.y + a * sin(ori * M_PI / 180.0);
        // p6.x = p.x - a * cos(ori * M_PI / 180.0);
        // p6.y = p.y - a * sin(ori * M_PI / 180.0);
    }
    else if (((B * p0.x + E) * (B * p0.x + E) - 4 * C * (A * p0.x * p0.x + D * p0.x + F)) > 0) // 垂直情况
    {
        vi = true;
        x1 = p0.x;
        x2 = p0.x;
        delta = pow((B * x1 + E), 2) - 4 * C * (A * x1 * x1 + D * x1 + F);
        // if (delta >= 0)
        // {
        // cout << "Situation2 intersection!!!" << endl;
        // }
        y1 = (-(B * x1 + E) - sqrt(delta)) / (2 * C);
        y2 = (-(B * x1 + E) + sqrt(delta)) / (2 * C);
        p1 = cv::Point2d(x1, y1);
        p2 = cv::Point2d(x2, y2);
        delta = pow((2 * B * E - 4 * C * D), 2) - 4 * (B * B - 4 * C * A) * (E * E - 4 * C * F);
        x3 = (-(2 * B * E - 4 * C * D) - sqrt(delta)) / (2 * (B * B - 4 * C * A));
        x4 = (-(2 * B * E - 4 * C * D) + sqrt(delta)) / (2 * (B * B - 4 * C * A));
        // cout << "delta3:" << pow((B * x3 + E), 2) - 4 * C * (A * x3 * x3 + D * x3 + F) << endl;
        // cout << "delta4:" << pow((B * x4 + E), 2) - 4 * C * (A * x4 * x4 + D * x4 + F) << endl;
        y3 = -(B * x3 + E) / (2 * C);
        y4 = -(B * x4 + E) / (2 * C);
        // cout << "x3:" << x3 << endl;
        // cout << "y3:" << y3 << endl;
        // cout << "x4:" << x4 << endl;
        // cout << "y4:" << y4 << endl;
        p3 = cv::Point2d(x3, y3);
        p4 = cv::Point2d(x4, y4);
        // p5.x = p.x + a * cos(ori * M_PI / 180.0);
        // p5.y = p.y + a * sin(ori * M_PI / 180.0);
        // p6.x = p.x - a * cos(ori * M_PI / 180.0);
        // p6.y = p.y - a * sin(ori * M_PI / 180.0);
    }
    if (p1.y < p2.y)
    {
        cv::Point2d temp;
        temp = p1;
        p1 = p2;
        p2 = temp;
    }
    if (p3.x > p4.x)
    {
        cv::Point2d temp;
        temp = p3;
        p3 = p4;
        p4 = temp;
    }

    // 按顺时针顺序存储
    pc.push_back(p0); // 像素坐标中心点
    pc.push_back(p1); // 圆下
    pc.push_back(p4); // 圆右
    pc.push_back(p2); // 圆上
    pc.push_back(p3); // 圆左

    pc.push_back(p4 - p0 + p2); // 圆右上
    pc.push_back(p4 - p0 + p1); // 圆右下
    pc.push_back(p3 - p0 + p1); // 圆左下
    pc.push_back(p3 - p0 + p2); // 圆左上

    // cout << p0.x << "," << p0.y << endl; // 中心
    // cout << p1.x << "," << p1.y << endl;
    // cout << p2.x << "," << p2.y << endl;
    // cout << p3.x << "," << p3.y << endl;
    // cout << p4.x << "," << p4.y << endl;
    // double result = A * p0.x * p0.x + B * p0.x * p0.y + C * p0.y * p0.y + D * p0.x + E * p0.y + F;
    // cout << "result:" << result << endl;
    // result = A * p1.x * p1.x + B * p1.x * p1.y + C * p1.y * p1.y + D * p1.x + E * p1.y + F;
    // cout << "result:" << result << endl;
    // result = A * p2.x * p2.x + B * p2.x * p2.y + C * p2.y * p2.y + D * p2.x + E * p2.y + F;
    // cout << "result:" << result << endl;
    // result = A * p3.x * p3.x + B * p3.x * p3.y + C * p3.y * p3.y + D * p3.x + E * p3.y + F;
    // cout << "result:" << result << endl;
    // result = A * p4.x * p4.x + B * p4.x * p4.y + C * p4.y * p4.y + D * p4.x + E * p4.y + F;
    // cout << "result:" << result << endl;
    // 20230914_xjl
    P0 = ep;
    double true_ellipse_d = 0.15;
    double dis_ = sqrt(dis * dis + true_ellipse_d * true_ellipse_d);
    double m = sqrt((fx + fy) * (fx + fy) / 4 + (p.x + a * cos(ori * M_PI / 180.0) - cx) * (p.x + a * cos(ori * M_PI / 180.0) - cx) + (p.y + a * sin(ori * M_PI / 180.0) - cy) * (p.y + a * sin(ori * M_PI / 180.0) - cy));
    double n = sqrt((fx + fy) * (fx + fy) / 4 + (p.x - a * cos(ori * M_PI / 180.0) - cx) * (p.x - a * cos(ori * M_PI / 180.0) - cx) + (p.y - a * sin(ori * M_PI / 180.0) - cy) * (p.y - a * sin(ori * M_PI / 180.0) - cy));
    // double cigma = asin(b / a) * 180 / M_PI;

    Pa1.x() = (p.x + a * cos(ori * M_PI / 180.0) - cx) * dis_ / m;
    Pa1.y() = (p.y + a * sin(ori * M_PI / 180.0) - cy) * dis_ / m;
    Pa1.z() = (fx + fy) / 2 * dis_ / m;
    Pa2.x() = (p.x - a * cos(ori * M_PI / 180.0) - cx) * dis_ / n;
    Pa2.y() = (p.y - a * sin(ori * M_PI / 180.0) - cy) * dis_ / n;
    Pa2.z() = (fx + fy) / 2 * dis_ / n;
    Vector3d fa;
    if (Pa1.x() > Pa2.x())
    {
        fa = g_.cross(Pa1 - Pa2);
    }
    else
    {
        fa = g_.cross(Pa2 - Pa1);
    }
    // cout << "Pa1-Pa2" << (Pa1 - Pa2).norm() << endl;
    // cout << "2a" << 2 * a << endl;
    P1 = ep - g_ * true_ellipse_d;
    P2 = ep + g_ * true_ellipse_d;
    fa.normalize();
    P3 = ep + true_ellipse_d * fa.cross(g_);
    P4 = ep - true_ellipse_d * fa.cross(g_);
    P5 = ep + fa * 0.15;
    if (P1.y() < P2.y())
    {
        Vector3d temp;
        temp = P1;
        P1 = P2;
        P2 = temp;
    }
    if (P3.x() > P4.x())
    {
        Vector3d temp;
        temp = P3;
        P3 = P4;
        P4 = temp;
    }
    Vector3d n_P5 = Rs[index] * (ric[0] * P5 + tic[0]) + Ps[index]; // 世界坐标系
    N_ = Rs[index] * ric[0] * fa;
    // 附加条件按顺序存储
    ps.push_back(P0); // 相机系下的坐标
    ps.push_back(P1);
    ps.push_back(P2);
    ps.push_back(P3);
    ps.push_back(P4);
    P5 = intrinsic_ * P5;
    ps.push_back(P5);
    P5 = P5 / P5.z();
    // pc.push_back(cv::Point2d(P5.x(), P5.y()));

    // cout << "P0" << P0 << endl;
    // cout << "P1" << P1 << endl;
    // cout << "P2" << P2 << endl;
    // cout << "P3" << P3 << endl;
    // cout << "P4" << P4 << endl;
    Vector3d n_P0 = Rs[index] * (ric[0] * P0 + tic[0]) + Ps[index];
    Vector3d n_P1 = Rs[index] * (ric[0] * P1 + tic[0]) + Ps[index];
    Vector3d n_P2 = Rs[index] * (ric[0] * P2 + tic[0]) + Ps[index];
    Vector3d n_P3 = Rs[index] * (ric[0] * P3 + tic[0]) + Ps[index];
    Vector3d n_P4 = Rs[index] * (ric[0] * P4 + tic[0]) + Ps[index];
    pw.push_back(n_P0);
    pw.push_back(n_P2);
    pw.push_back(n_P4);
    pw.push_back(n_P1);
    pw.push_back(n_P3);
    pw.push_back(n_P5);

    // ps.push_back(p_P0);
    // ps.push_back(p_P1);
    // ps.push_back(p_P2);
    // ps.push_back(p_P3);
    // ps.push_back(p_P4);
    // A = (cos(theta) / a) * (cos(theta) / a) + (sin(theta) / b) * (sin(theta) / b);
    // B = 2 * (1 / (a * a) - 1 / (b * b)) * sin(theta) * cos(theta);
    // C = (sin(theta) / a) * (sin(theta) / a) + (cos(theta) / b) * (cos(theta) / b);
    // D = -2 * A * p_P0.x() - B * p_P0.y();
    // E = -B * p_P0.x() - 2 * C * p_P0.y();
    // F = A * p_P0.x() * p_P0.x() + B * p_P0.x() * p_P0.y() + C * p_P0.y() * p_P0.y() - 1;
    // result = A * p_P0.x() * p_P0.x() + B * p_P0.x() * p_P0.y() + C * p_P0.y() * p_P0.y() + D * p_P0.x() + E * p_P0.y() + F;
    // cout << "result:" << result << endl;
    // result = A * p_P1.x() * p_P1.x() + B * p_P1.x() * p_P1.y() + C * p_P1.y() * p_P1.y() + D * p_P1.x() + E * p_P1.y() + F;
    // cout << "result:" << result << endl;
    // result = A * p_P2.x() * p_P2.x() + B * p_P2.x() * p_P2.y() + C * p_P2.y() * p_P2.y() + D * p_P2.x() + E * p_P2.y() + F;
    // cout << "result:" << result << endl;
    // result = A * p_P3.x() * p_P3.x() + B * p_P3.x() * p_P3.y() + C * p_P3.y() * p_P3.y() + D * p_P3.x() + E * p_P3.y() + F;
    // cout << "result:" << result << endl;
    // result = A * p_P4.x() * p_P4.x() + B * p_P4.x() * p_P4.y() + C * p_P4.y() * p_P4.y() + D * p_P4.x() + E * p_P4.y() + F;
    // cout << "result:" << result << endl;
    return;
}

bool Estimator::choose_right(vector<cv::Point2d> input_pts1, vector<cv::Point2d> input_pts2)
{
    double k1 = (input_pts1[1].y - input_pts1[2].y) / (input_pts1[1].x - input_pts1[2].x);
    double k2 = (input_pts2[1].y - input_pts2[2].y) / (input_pts2[1].x - input_pts2[2].x);
    cv::Point2d middle1, middle2;
    middle1 = (input_pts1[1] + input_pts1[3]) / 2.0;
    middle2 = (input_pts2[1] + input_pts2[3]) / 2.0;
    // if (cv::norm(middle1 - input_pts1[0]) > 5)
    // {
    //     cout << "111:" << cv::norm(middle1 - input_pts1[0]) << endl;
    //     return false;
    // }
    // if (cv::norm(middle2 - input_pts2[0]) > 5)
    // {
    //     cout << "222:" << cv::norm(middle2 - input_pts2[0]) << endl;
    //     return false;
    // }
    if (k1 * k2 < -0.001)
    {
        return false;
    }
    return true;
}

double Estimator::computeTri(vector<cv::Point2d> &p, Vector3d &ep, Vector3d &np, Matrix3d intrinsic, double trued, int index, Vector3d g)
{
    double sum, dis, average;
    cv::Point2d p0, p1;
    for (int i = 0; i < p.size(); i++)
    {
        p0 = p[i];
        if (i == p.size() - 1)
            sum += sqrt((p0.x - p[0].x) * (p0.x - p[0].x) + (p0.y - p[0].y) * (p0.y - p[0].y));
        else
            sum += sqrt((p0.x - p[i + 1].x) * (p0.x - p[i + 1].x) + (p0.y - p[i + 1].y) * (p0.y - p[i + 1].y));
        p1.x += p[i].x;
        p1.y += p[i].y;
    }
    average = sum / 3;
    average = 2 * average / (intrinsic(0, 0) + intrinsic(1, 1));
    dis = trued / average;
    p1.x = p1.x / 3.0;
    p1.y = p1.y / 3.0; // 求中心点
    double u = (p1.x - intrinsic(0, 2)) / intrinsic(0, 0);
    double v = (p1.y - intrinsic(1, 2)) / intrinsic(1, 1);
    ep.x() = u * dis;
    ep.y() = v * dis;
    ep.z() = dis;
    np = Rs[index] * (ric[0] * ep + tic[0]) + Ps[index];

    cv::Point2d p3, p4;
    p3.x = p[2].x - g[0] * 5;
    p3.y = p[2].y - g[1] * 5;
    double k = (p3.y - p[2].y) / (p3.x - p[2].x);
    double b = p3.y - k * p3.x;
    double m = (p[1].y - p[3].y) / (p[1].x - p[3].x);
    double n = p[1].y - m * p[1].x;
    p4.x = (p[1].y - b - m * p[1].x) / (k - m);
    p4.y = k * p4.x + b;
    p.push_back(p4);
    return dis;
}

// 20221213 xjl
// 输入：路标时间戳、状态列表、路标的相机系坐标，输出世界系坐标
//  bool sign2local(double time, std::queue<StateData> &statelist, Eigen::Vector3d c_pos_ellipse, Eigen::Vector3d &n_pos_ellipse)
bool Estimator::sign2local(double time, int &index, Eigen::Vector3d c_pos_ellipse, Eigen::Vector3d &n_pos_ellipse)
{
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
    Eigen::Matrix3d rot = Matrix3d::Identity();
    Eigen::Vector3d p;
    double time_[WINDOW_SIZE];
    index = -1;
    Matrix3d r;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        time_[i] = Headers[i].stamp.toSec();
    }
    for (int j = 0; j < WINDOW_SIZE; j++)
    {
        if (abs(time - time_[j]) < 0.001)
        {
            index = j;
            break;
        }
        else
        {
            index = -1;
        }
    }
    if (time > time_[WINDOW_SIZE - 1]) // 若图像帧还没来
    {
        index = WINDOW_SIZE - 1;
        n_pos_ellipse = Rs[WINDOW_SIZE] * (ric[0] * c_pos_ellipse + tic[0]) + Ps[WINDOW_SIZE];
        printf("\033[1;32m  sign time: %f \n\033[0m", time);
        printf("\033[1;33m  last camera time: %f \n\033[0m", time_[index]);
        printf("\033[1;34m  camera time begin at : %f  to  %f \n\033[0m", time_[0], time_[WINDOW_SIZE]);
        return true;
    }
    else if (index == -1) // 图像来了但错过了
    {
        // p.x() = Ps[WINDOW_SIZE].x();
        // p.y() = Ps[WINDOW_SIZE].y();
        // p.z() = Ps[WINDOW_SIZE].z();
        // r = ric[0];
        // t = tic[0];
        // q = Quaterniond(Rs[WINDOW_SIZE]);
        // q.norm();
        // rot = q.toRotationMatrix();
        // std::cout << "position :" << Ps[WINDOW_SIZE] << std::endl;
        // std::cout << "yaw :" << Utility::R2ypr(Rs[WINDOW_SIZE]).x() << std::endl;
        // std::cout << "c_pos_ellipse :" << c_pos_ellipse << std::endl;
        // n_pos_ellipse = p + rot.transpose() * (r * (c_pos_ellipse - t));
        // n_pos_ellipse = Ps[WINDOW_SIZE] + Rs[WINDOW_SIZE].transpose() * c_pos_ellipse;
        // std::cout << "R :" << rot * r << std::endl;
        // ROS_WARN("ellipse time: %f", time);
        ROS_WARN("no camera time matching!!!");
        printf("\033[1;32m  sign time: %f \n\033[0m", time);
        printf("\033[1;34m  camera time begin at : %f  to  %f \n\033[0m", time_[0], time_[WINDOW_SIZE - 1]);
        return false;
    }
    else // 找到了
    {
        // p.x() = Ps[index].x();
        n_pos_ellipse = Rs[index] * (ric[0] * c_pos_ellipse + tic[0]) + Ps[index];
        // p.y() = Ps[index].y();
        // p.z() = Ps[index].z();
        // r = ric[0];
        // t = tic[0];
        // q = Quaterniond(Rs[index]);
        // q.norm();
        // rot = q.toRotationMatrix();
        // printf("\033[1;32m  position: %f ,%f ,%f \n\033[0m", Ps[index].x(), Ps[index].y(), Ps[index].z());
        // printf("\033[1;33m  yaw: %f \n\033[0m", Utility::R2ypr(Rs[index]).x());
        // printf("\033[1;34m  c_pos_ellipse: %f, %f, %f \n\033[0m", c_pos_ellipse.x(), c_pos_ellipse.y(), c_pos_ellipse.z());

        // n_pos_ellipse = Ps[index] + Rs[index].transpose() * c_pos_ellipse;

        // std::cout << "position :" << Ps[index] << std::endl;
        // std::cout << "pose :" << Utility::R2ypr(Rs[index]).x() << std::endl;
        // std::cout << "c_pos_ellipse :" << c_pos_ellipse << std::endl;
        // n_pos_ellipse = p + rot.transpose() * (r * (c_pos_ellipse - t));
        // printf("\033[1;36mn_pos_ellipse:\n\033[0m");
        // printf("x:%f\n",n_pos_ellipse.x());
        // printf("y:%f\n",n_pos_ellipse.y());
        // printf("z:%f\n",n_pos_ellipse.z());
        // n_pos_ellipse.x() = Ps[index].x() + sin(Utility::R2ypr(Rs[index]).x()) * c_pos_ellipse.norm();
        // n_pos_ellipse.y() = Ps[index].y() + cos(Utility::R2ypr(Rs[index]).x()) * c_pos_ellipse.norm();
        // n_pos_ellipse.z() = Ps[index].y() + c_pos_ellipse.z();
        printf("\033[1;32m  sign time: %f \n\033[0m", time);
        printf("\033[1;33m  camera time: %f \n\033[0m", time_[index]);
        printf("\033[1;34m  camera time begin at : %f  to  %f \n\033[0m", time_[0], time_[WINDOW_SIZE - 1]);
        return true;
    }
}

bool Estimator::timeup(double ellipse_detec_time)
{
    if (ellipse_detec_time < Headers[WINDOW_SIZE - 1].stamp.toSec())
    {
        return true;
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
    // VectorXd dep = f_manager.getDepthVector();
    // for (int i = 0; i < map_manager.getFeatureCount(); i++)
    //     para_sign_Pose[i][0] = dep(i);
}

void Estimator::double2vector_()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = Vector3d(para_Pose[i][0],
                         para_Pose[i][1],
                         para_Pose[i][2]);
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]); // 优化前的第一帧的姿态和位置进行锁定
    Vector3d origin_P0 = Ps[0];
    // xjl
    int counter = 0;
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        if (sign_in_window[i])
        {
            counter++;
        }
    }
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0); // 上一次优化的姿态
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix()); // 优化之后的第一帧姿态
    double y_diff = origin_R0.x() - origin_R00.x();                // yaw角偏离多少
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0)); // 偏离的yaw角要进行补偿，转为旋转矩阵
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = (rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                     para_Pose[i][1] - para_Pose[0][1],
                                     para_Pose[i][2] - para_Pose[0][2]) +
                 origin_P0);
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        // cout << "vins relo " << endl;
        // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }

    // add sign reloc xjl
    if (counter >= 2)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(para_Pose[2][6], para_Pose[2][3], para_Pose[2][4], para_Pose[2][5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(para_Pose[2][0] - para_Pose[0][0],
                                     para_Pose[2][1] - para_Pose[0][1],
                                     para_Pose[2][2] - para_Pose[0][2]) +
                 origin_P0;

        Matrix3d sign_relo_r = Quaterniond(para_Pose[2][6], para_Pose[2][3], para_Pose[2][4], para_Pose[2][5]).normalized().toRotationMatrix();
        Vector3d sign_relo_t = Vector3d(para_Pose[2][0], para_Pose[2][1], para_Pose[2][2]);
        double drift_correct_yaw_sign;
        drift_correct_yaw_sign = Utility::R2ypr(sign_relo_r).x() - Utility::R2ypr(relo_r).x(); // 之前的减去现在的
        // if (drift_correct_yaw_sign > 0.1 || (relo_t - sign_relo_t).norm() > 0.1)
        if ((relo_t - sign_relo_t).norm() > 0.1)
        {
            setSignFrame(Headers[0].stamp.toSec());
            drift_sign_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw_sign, 0, 0));
            drift_sign_correct_t = sign_relo_t - relo_t; // 之前的减去yaw纠正后的现在的
            sign_relative_t = drift_sign_correct_t;
            sign_relative_q = drift_sign_correct_r;
            sign_relative_yaw = drift_correct_yaw_sign;
            // sign_relative_t = sign_relo_r.transpose() * (Ps[5] - sign_relo_t);
            // sign_relative_q = sign_relo_r.transpose() * Rs[5];
            // sign_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Quaterniond(para_Pose[5][6], para_Pose[5][3], para_Pose[5][4], para_Pose[5][5]).normalized().toRotationMatrix()).x() - Utility::R2ypr(relo_r).x());
            sign_frame_index = 2;
            cout << "sign_relative_t" << sign_relative_t << endl;
            cout << "sign_relative_q" << sign_relative_q.w() << sign_relative_q.x() << sign_relative_q.y() << sign_relative_q.z() << endl;
        }
        // if (sign_relative_yaw < 0.01 && sign_relative_t.norm() < 0.1)
        Matrix3d sign_r;
        Vector3d sign_t;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            double drift_sign_correct_yaw;
            sign_r = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            sign_t = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
            drift_sign_correct_yaw = Utility::R2ypr(sign_r).x() - Utility::R2ypr(Rs[i]).x();
            // sign_relative_yaw = drift_sign_correct_yaw;
            // Rs[i] = Utility::ypr2R(Vector3d(drift_sign_correct_yaw, 0, 0)) * Rs[i];
            // Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            // Ps[i] = sign_relative_t + Ps[i];
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
            Vs[i] = Vector3d(para_SpeedBias[i][0],
                             para_SpeedBias[i][1],
                             para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        // return true;
    }
    return false;
}

void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    ceres::LossFunction *loss_function1;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    loss_function1 = new ceres::TrivialLoss;
    vector2double(); // 将系统状态向量转为double类型数组以适应ceres

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // 优化P和Q
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);               // 优化速度和零偏V ba bg
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); // 优化外参ric,tic
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1); // 优化时间延迟
        // problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    // vector2double(); // 将系统状态向量转为double类型数组以适应ceres

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks); // 边缘化的残差块
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);                                                   // 预积分
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]); // IMU因子残差块，可以看到有前后帧的联系
    }

    int s_m_cnt = 0;
    addsign = false;
    // printf("window time begin at %f to %f \n", Headers[0].stamp.toSec(), Headers[WINDOW_SIZE - 1].stamp.toSec());
    for (auto &it_per_id : map_manager.sign) // 遍历滑窗内的标志
    {
        if (it_per_id.used_num > 2 && !it_per_id.sign_per_frame.empty())
        {
            int frame_cnt = 0;
            for (auto &it_per_frame : it_per_id.sign_per_frame) // 该标志在滑窗内每一帧（可能不是）的信息
            {
                SIGN reds_sign;
                reds_sign.signclass = it_per_id.classify;
                reds_sign.N = it_per_id.N_;
                reds_sign.C = it_per_id.C_;
                reds_sign.cvPoints = it_per_frame.pts;
                reds_sign.time = it_per_frame.time;
                reds_sign.scale = 0.3;
                for (int i = 0; i < WINDOW_SIZE; i++)
                {
                    if (isTheSameTimeNode(reds_sign.time, Headers[i].stamp.toSec(), MINIMUM_TIME_INTERVAL))
                    {
                        sign_in_window[i] = true;
                        problem.SetParameterBlockConstant(para_SpeedBias[i]);
                        frame_cnt = i;
                        if (ESTIMATE_SIGN && frame_cnt != 0)
                        {
                            if (reds_sign.signclass[0] == 'c')
                            {
                                addsign = true;
                                printf("add circle residual at %f, sign id is: %d \n", reds_sign.time, it_per_id.sign_id);
                                // NEWSIGNFactor *newsign_factor = new NEWSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(newsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNFactor, 18, 7, 7>(new SIGNFactor(reds_sign)), loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // printf("%f", reds_sign.time);
                                // printf("%f", Headers[frame_cnt].stamp.toSec());
                                // double **para = new double *[3];
                                // para[0] = para_Pose[frame_cnt];
                                // para[1] = para_Ex_Pose[0];
                                // para[2] = para_sign_Pose[id];
                                // newsign_factor->check(para);
                            }
                            else if (reds_sign.signclass[0] == 'r')
                            {
                                addsign = true;
                                printf("add rectangle residual at %f, sign id is: %d \n", reds_sign.time, it_per_id.sign_id);
                                // NEWRSIGNFactor *rsign_factor = new NEWRSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(rsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNRFactor, 18, 7, 7>(new SIGNRFactor(reds_sign)), loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                            }
                            else if (reds_sign.signclass[0] == 't')
                            {
                                addsign = true;
                                printf("add triangle residual at %f, sign id is: %d \n", reds_sign.time, it_per_id.sign_id);
                                // NEWTSIGNFactor *tsign_factor = new NEWTSIGNFactor(para_Pose[frame_cnt], para_Ex_Pose[0], reds_sign);
                                // problem.AddResidualBlock(tsign_factor, loss_function1, para_Pose[frame_cnt], para_Ex_Pose[0]);
                                // problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SIGNTFactor, 10, 7, 7>(new SIGNTFactor(reds_sign)), loss_function, para_Pose[frame_cnt], para_Ex_Pose[0]);
                            }
                        }
                    }
                }
            }
            // ROS_INFO("sign measurement count: %d", frame_cnt);
        }
        s_m_cnt++;
        // frame_cnt++;
    }
    // if (!RT_from_signs.empty())
    // {
    //     for (RT rt : RT_from_signs)
    //     {
    //         for (size_t k = 0; k < WINDOW_SIZE; k++)
    //         {
    //             if (isTheSameTimeNode(rt.time, Headers[k].stamp.toSec(), MINIMUM_TIME_INTERVAL))
    //             {
    //                 cout << "abs sign add!!!" << endl;
    //                 Quaterniond q = Quaterniond(rt.R);
    //                 double p[7];
    //                 p[0] = rt.T.x();
    //                 p[1] = rt.T.y();
    //                 p[2] = rt.T.z();
    //                 p[3] = q.x();
    //                 p[4] = q.y();
    //                 p[5] = q.z();
    //                 p[6] = q.w();
    //                 problem.AddResidualBlock(new ceres::AutoDiffCostFunction<MultisignFactor, 7, 7>(new MultisignFactor(para_Pose[k], p)), loss_function1, para_Pose[k]);
    //             }
    //         }
    //     }
    // }
    // add sign factor

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 地图点坐标

        for (auto &it_per_frame : it_per_id.feature_per_frame) // 该地图点在每一帧的信息
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point; // 归一化图像坐标
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                  it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]); // 带时延估计的重投影残差
                /*
                double **para = new double *[5];
                para[0] = para_Pose[imu_i];
                para[1] = para_Pose[imu_j];
                para[2] = para_Ex_Pose[0];
                para[3] = para_Feature[feature_index];
                para[4] = para_Td[0];
                f_td->check(para);
                */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]); // 不带时延估计的重投影残差
            }
            f_m_cnt++;
        }
    }

    if (relocalization_info) // 如果有设置回环
    {
        // printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization); // 输入了回环帧的pose
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index) // 若该特征点起始帧在回环帧之前
            {
                while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id) // 若匹配点的z
                {
                    retrive_feature_index++;
                }
                if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // 线性求解
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS; // 最大迭代次数
    // if (addsign == true)
    //     options.max_num_iterations = NUM_ITERATIONS * 2; // 最大迭代次数
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << endl;

    // cout << summary.FullReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) // 滑掉老帧，进行边缘化
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                    continue;
                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
    double time[WINDOW_SIZE];
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        time[i] = Headers[i].stamp.toSec();
    }
    map_manager.updateSign(time); // 更新标志列表
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
    // map_manager.removesignFront();
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
        // map_manager.removesignBack();
        // map_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
    {
        f_manager.removeBack();
        // map_manager.removesignBack();
    }
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

void Estimator::setSignFrame(double stamp)
{
    sign_frame_stamp = stamp;
}