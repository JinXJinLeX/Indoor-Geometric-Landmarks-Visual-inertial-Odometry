#include "feature_tracker.h"

int FeatureTracker::n_id = 0;
// 20230223xjl
int TRACK_PYRAMID_LEVEL = 3;
bool first_imu_ = 1;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

cv::Point3f FeatureTracker::pixel2cam(const cv::Point2f pts)
{
    // Lift points to normalised plane 归一化平面
    Eigen::Vector3d tmp_p;
    m_camera->liftProjective(Eigen::Vector2d(pts.x, pts.y), tmp_p);
    // tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
    // tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
    // un_cur_pts = cv::Point2f(tmp_p.x(), tmp_p.y());
    // readFromYamlFile
    // double y = (pts.y - m_cy) / m_fy;
    // double x = (pts.x - m_cx - skew * y) / m_fx;
    return {tmp_p.x(), tmp_p.y(), 1.0};
}

cv::Point2f FeatureTracker::cam2pixel(const cv::Point3f pts)
{
    Eigen::Vector2d p;
    Eigen::Vector3d pt;
    pt.x() = pts.x;
    pt.y() = pts.y;
    pt.z() = pts.z;
    m_camera->spaceToPlane(pt, p);
    return {p.x(), p.y()};
}

// void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, queue<IMU> &ins_window)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    Eigen::Matrix3f r_cur_pre;
    Eigen::Vector3f t_cur_pre;
    r_cur_pre = get_ins(ins_window, cur_time, prev_time);
    cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 952.011, 0.000000, 661.733, 0.000000, 959.650, 375.598, 0, 0, 1);
    cv::Mat distortion;
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 892.982544, 0.000000, 611.368633, 0.000000, 900.910278, 353.958833, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0,0);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 619.523712, 0.000000, 656.497684, 0.000000, 615.410395, 403.222400, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(4, 1) << -0.049547, 0.012867, -0.000750, -0.000176);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 387.240631, 0.000000, 321.687063, 0.000000, 387.311676, 251.179550, 0, 0, 1);
    // cv::Mat distortion = (cv::Mat_<double>(1, 4) << -0.055379, 0.051226, 0.000408, -0.002483);
    // cv::Mat distortion = (cv::Mat_<double>(4, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
    // cv::Mat intrinsic = (cv::Mat_<double>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
    vector<Vector2d> pc_predict;       // 预测特征点
    vector<cv::Point2f> un_pc_predict; // 去畸变后的预测特征点

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
        // 20230220xjl
        if (cur_pts.size() > 0) // cur_pts指的是上一帧的特征点(Point2f)
        {
            vector<cv::Point2f> un_pts;
            // cv::Point3d pc_pre;
            TicToc t_c;
            // 此处要加入一个对cur_pts去畸变的函数
            // cv::undistortPoints(cur_pts, un_pts, intrinsic, distortion, cv::Mat(), intrinsic);
            un_pts=cur_pts;
            int j = 0;
            for (auto pts : un_pts)
            {
                // 返回归一化坐标
                cv::Point3f pc_pre = FeatureTracker::pixel2cam(pts);
                Eigen::Vector3f pc_cur;
                // 归一化坐标预测下一帧特征点的2d位置
                pc_cur.x() = pc_pre.x;
                pc_cur.y() = pc_pre.y;
                pc_cur.z() = pc_pre.z;
                pc_cur = r_cur_pre * pc_cur;
                // pc_cur.normalize();
                // 对每个点归一化
                cv::Point3f pcc;
                // pcc.x = pc_cur.x() / pc_cur.z();
                // pcc.y = pc_cur.y() / pc_cur.z();
                // pcc.z = 1.0;
                pcc.x = pc_cur.x();
                pcc.y = pc_cur.y();
                pcc.z = pc_cur.z();

                // 20230223xjl
                cv::Point2f predict_point = FeatureTracker::cam2pixel(pcc);
                j++;
                un_pc_predict.emplace_back(predict_point);
                // pc_predict.push_back(b);
                // 灰度处理函数，输入为两帧的图像、前一帧特征点和后一帧的预测点
            }
            // 投影函数，将前帧归一化点投影到当前帧
            // m_camera->projectPoints(pc_cur_vec, r_cur_pre, t_cur_pre, un_pc_predict); // 此处的预测点还没有加畸变
            // ROS_WARN("***********!!!!!!!!!!!!!!!!!!!***************");
            // ROS_WARN("pior costs: %fms", t_c.toc()); // 可以看到在不做任何处理的情况下仅仅输出惯性先验，只需要不到1ms时间
            un_pts.clear();
        }
        // forw_img = img;
    }

    forw_pts.clear();
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, un_pc_predict, status,
                                 err, cv::Size(31, 31), TRACK_PYRAMID_LEVEL,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW); // 与当前帧进行光流
        forw_pts = un_pc_predict;
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        pc_predict.clear();
        un_pc_predict.clear();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

// 20230227xjl
// 欧拉角转旋转矩阵
Eigen::Matrix3d euler2Rotation(Eigen::Vector3d &theta)
{
    // Calculate rotation about x axis
    Eigen::Matrix3d R_x;
    R_x << 1, 0, 0,
        0, cos(theta[0]), -sin(theta[0]),
        0, sin(theta[0]), cos(theta[0]);
    // Calculate rotation about y axis
    Eigen::Matrix3d R_y;
    R_y << cos(theta[1]), 0, sin(theta[1]),
        0, 1, 0,
        -sin(theta[1]), 0, cos(theta[1]);
    // // Calculate rotation about z axis
    Eigen::Matrix3d R_z;
    R_z << cos(theta[2]), -sin(theta[2]), 0,
        sin(theta[2]), cos(theta[2]), 0,
        0, 0, 1;
    // // Combined rotation matrix
    Eigen::Matrix3d R = R_z * R_y * R_x;
    return R;
}

// IMU积分
void integrationProcess(queue<IMU> &ins_series, Matrix3f &R)
{
    IMU imu_pre;
    IMU imu_cur;
    Eigen::Vector3f gyr_0;
    Eigen::Vector3d delta_Q;
    Eigen::Quaterniond q;
    double dt;

    while (!ins_series.empty())
    {
        if (first_imu_)
        {
            imu_pre = ins_series.front();
            ins_series.pop();
            first_imu_ = false;
            delta_Q = imu_pre.dtheta;
        }
        imu_cur = ins_series.front();
        dt = imu_cur.time;
        delta_Q += imu_cur.dtheta;
        ins_series.pop();
    };
    Eigen::Matrix3d R_ = euler2Rotation(delta_Q);
    // std::cout << "delta_Q:"<<delta_Q<<std::endl;
    // std::cout << "R_" <<R_ <<std::endl;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R(i, j) = (float)(R_(i, j));
        }
    }
    first_imu_ = true;
}
Matrix3f FeatureTracker::get_ins(queue<IMU> &ins_window, double cur_time, double prev_time)
{
    Matrix3f R;
    R << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    if (cur_time == 0 || prev_time == 0)
        return R;
    IMU imu;
    double time;
    queue<IMU> ins_series;
    double cur_image_time = cur_time;
    double pre_image_time = prev_time;
    while (!ins_window.empty())
    {
        imu = ins_window.front();
        time = imu.time;
        // 在前一帧图像时间戳之前的imu帧要丢掉
        if (time < pre_image_time)
            ins_window.pop();
        // 在两帧图像时间戳之间的imu帧要保存
        else if (time >= pre_image_time && time < cur_image_time)
        {
            ins_window.pop();
            ins_series.emplace(imu);
        }
        // 在后一帧图像时间戳之后的imu帧不动
        else if (time >= cur_image_time)
            break;
    }
    integrationProcess(ins_series, R); // IMU积分
    return R;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
