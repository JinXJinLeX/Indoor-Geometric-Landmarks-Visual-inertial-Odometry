#include "new_sign_factor.h"

Eigen::MatrixXd NEWSIGNFactor::sqrt_info;
double NEWSIGNFactor::sum_t;
Eigen::MatrixXd NEWRSIGNFactor::sqrt_rinfo;
double NEWRSIGNFactor::sum_t;
Eigen::MatrixXd NEWTSIGNFactor::sqrt_tinfo;
double NEWTSIGNFactor::sum_t;

NEWSIGNFactor::NEWSIGNFactor(const double p[7], const double p_[7], const SIGN sign_)
{
    pose[7] = p[7];
    ex_pose[7] = p_[7];
    sign = sign_;
};

bool NEWSIGNFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;                                                          // 耗时
    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]); // 载体body
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]); // 外参
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]); // 标志位置
    // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    // std::cout << "P" << P.x() << P.y() << P.z() << std::endl;
    // std::cout << "Q" << Q.x() << Q.y() << Q.z() << Q.w() << std::endl;

    // double inv_dep = parameters[2][2];
    bool change1, change2;
    change1 = false;
    change2 = false;
    std::vector<Eigen::Vector3d> pts_imu;                                                                              // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera;                                                                           // 关键点相机系下坐标
    pts_imu.push_back(Q.inverse() * (sign.C - P));                                                                     // 中
    pts_imu.push_back(Q.inverse() * (sign.C - (0.5 * sign.scale / G.norm()) * G - P));                                 // 下
    pts_imu.push_back(Q.inverse() * (sign.C - (0.5 * sign.scale / (sign.N.cross(G)).norm()) * (sign.N.cross(G)) - P)); // 右
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale / G.norm() * G - P));                                   // 上
    pts_imu.push_back(Q.inverse() * (sign.C + (0.5 * sign.scale / (sign.N.cross(G)).norm()) * (sign.N.cross(G)) - P)); // 左
    double inv_dep[5];

    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera.push_back(qic.inverse() * (pts_imu[i] - tic));
    }
    Eigen::Map<Eigen::Matrix<double, 10, 1>> residual(residuals);
    for (int i = 0; i < pts_imu.size(); i++)
    {
        inv_dep[i] = (pts_camera[i].z());
        residual.block<2, 1>(2 * i, 0) = (pts_camera[i] / inv_dep[i]).head<2>() - sign.cvPoints[i];
        // cout << "ProPoints:" << (pts_camera[i] / inv_dep[i]).x() << "," << (pts_camera[i] / inv_dep[i]).y() << endl;
        // cout << "cvPoints:" << sign.cvPoints[i].x() << "," << sign.cvPoints[i].y() << endl;
    }
    residual = sqrt_info * residual;
    // cout << "原始圆形残差:" << residual.transpose() << endl;
    if (jacobians)
    {
        Eigen::Matrix3d R = Q.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 10, 15> reduce(10, 15);
        reduce.setZero();
        for (int i = 0; i < pts_camera.size(); i++)
        {
            reduce.block<2, 3>(2 * i, 3 * i) << 1.0 / inv_dep[i], 0, -pts_camera[i](0) / (inv_dep[i] * inv_dep[i]),
                0, 1.0 / inv_dep[i], -pts_camera[i](1) / (inv_dep[i] * inv_dep[i]);
        }
        // cout << reduce << endl;
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double, 15, 6> jaco;
            // 圆心
            jaco.block<3, 3>(0, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - P))); // 对Q求导
            jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[0]); // 对Q求导
            // 圆下
            jaco.block<3, 3>(3, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - sign.scale * (G / G.norm()) - P))); // 对Q求导
            jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[1]); // 对Q求导
            // 圆右
            jaco.block<3, 3>(6, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[2]); // 对Q求导
            // 圆上
            jaco.block<3, 3>(9, 0) = ric.transpose() * -R.transpose();                     // 对P求导
            jaco.block<3, 3>(9, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[3]); // 对Q求导
            // 圆左
            jaco.block<3, 3>(12, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[4]); // 对Q求导

            // jaco.leftCols<3>() = -ric.transpose() * R.transpose();
            // jaco.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(R.transpose() * (C - P));

            jacobian_pose.leftCols<6>() = reduce * (jaco);
            jacobian_pose.rightCols<1>().setZero();
            // cout << jacobian_pose << endl;
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            Eigen::Matrix<double, 15, 6> jaco_ex;
            // 圆心
            jaco_ex.block<3, 3>(0, 0) = -ric.transpose(); // 对tic求导
            // jaco_ex.block<3, 3>(0, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - P) - tic)); // 对ric求导
            // 圆下
            jaco_ex.block<3, 3>(3, 0) = -ric.transpose(); // 对tic求导
            // jaco_ex.block<3, 3>(3, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - sign.scale * G / G.norm() - P) - tic)); // 对ric求导
            // 圆右
            jaco_ex.block<3, 3>(6, 0) = -ric.transpose(); // 对tic求导
            // jaco_ex.block<3, 3>(6, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P) - tic)); // 对ric求导
            // 圆上
            jaco_ex.block<3, 3>(9, 0) = -ric.transpose(); // 对tic求导
            // jaco_ex.block<3, 3>(9, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C + sign.scale * G / G.norm() - P) - tic)); // 对ric求导
            // 圆左
            jaco_ex.block<3, 3>(12, 0) = -ric.transpose(); // 对tic求导
            // jaco_ex.block<3, 3>(12, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C + sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P) - tic)); // 对ric求导

            jacobian_ex_pose.leftCols<6>() = reduce * (jaco_ex);
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // if (jacobians[2])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 10, 6, Eigen::RowMajor>> jacobian_sign_pose(jacobians[2]);
        //     Eigen::Matrix<double, 15, 6> jaco_sign_pose;
        //     Eigen::Matrix3d I;
        //     I.setIdentity();
        //     Eigen::Matrix3d Temp = I - ((N.cross(G)) / ((N.cross(G)).norm())) * (((N.cross(G)) / ((N.cross(G)).norm())).transpose()) / (N.cross(G)).norm();
        //     // 圆心
        //     jaco_sign_pose.block<3, 3>(0, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(0, 3).setZero();                         // 对N求导
        //     // 圆上
        //     jaco_sign_pose.block<3, 3>(3, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(3, 3).setZero();                         // 对N求导
        //     // 圆右
        //     jaco_sign_pose.block<3, 3>(6, 0) = ric.transpose() * R.transpose();                                                   // 对C求导
        //     jaco_sign_pose.block<3, 3>(6, 3) = -ric.transpose() * R.transpose() * sign.scale * Temp * Utility::skewSymmetric(-G); // 对N求导
        //     // 圆下
        //     jaco_sign_pose.block<3, 3>(9, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(9, 3).setZero();                         // 对N求导
        //     // 圆左
        //     jaco_sign_pose.block<3, 3>(12, 0) = ric.transpose() * R.transpose();                                                  // 对C求导
        //     jaco_sign_pose.block<3, 3>(12, 3) = ric.transpose() * R.transpose() * sign.scale * Temp * Utility::skewSymmetric(-G); // 对N求导
        //     // jaco_sign_pose.leftCols<3>() = ric.transpose() * R.transpose();
        //     // jaco_sign_pose.rightCols<3>().setZero();

        //     jacobian_sign_pose = 0.3 * reduce * (jaco_sign_pose);
        // }
    }
    sum_t += tic_toc.toc();
    return true;
}

void NEWSIGNFactor::check(double **parameters)
{
    double *res = new double[10];
    double **jaco = new double *[3];
    jaco[0] = new double[10 * 7];
    jaco[1] = new double[10 * 7];
    // jaco[2] = new double[10 * 6];

    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 10, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    // std::cout << Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
    //           << std::endl;
    // std::cout << Eigen::Map<Eigen::Matrix<double, 10, 6, Eigen::RowMajor>>(jaco[2]) << std::endl
    //           << std::endl;

    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
    bool change1 = false;
    bool change2 = false;
    pts_imu.push_back(Q.inverse() * (C - P));
    pts_imu.push_back(Q.inverse() * (C - sign.scale * G / G.norm() - P));
    pts_imu.push_back(Q.inverse() * (C - sign.scale * (N.cross(G)) / ((N.cross(G)).norm()) - P));
    pts_imu.push_back(Q.inverse() * (C + sign.scale * G / G.norm() - P));
    pts_imu.push_back(Q.inverse() * (C + sign.scale * (N.cross(G)) / ((N.cross(G)).norm()) - P));

    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera.push_back(qic.inverse() * (pts_imu[i] - tic));
    }
    // if (pts_camera[1].y() < pts_camera[3].y())
    // {
    //     change1 = true;
    //     Eigen::Vector3d temp;
    //     temp = pts_camera[1];
    //     pts_camera[1] = pts_camera[3];
    //     pts_camera[3] = temp;
    // }
    // if (pts_camera[2].x() < pts_camera[4].x())
    // {
    //     change2 = true;
    //     Eigen::Vector3d temp;
    //     temp = pts_camera[2];
    //     pts_camera[2] = pts_camera[4];
    //     pts_camera[4] = temp;
    // }

    Eigen::Matrix<double, 10, 1> residual;
    for (int i = 0; i < pts_imu.size(); i++)
    {
        residual.block<2, 1>(2 * i, 0) = (pts_camera[i] / pts_camera[i].z()).head<2>() - sign.cvPoints[i];
    }

    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 0.2;
    // const double eps = 1e-2;
    {
        Eigen::Matrix<double, 10, 19> num_jacobian;
        Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);
        std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
        std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
        for (int k = 0; k < 19; k++)
        {
            pts_camera.clear();
            pts_imu.clear();
            pts_imu.push_back(Q.inverse() * (C - P));
            pts_imu.push_back(Q.inverse() * (C - sign.scale * G / G.norm() - P));
            pts_imu.push_back(Q.inverse() * (C - sign.scale * (N.cross(G)) / ((N.cross(G)).norm()) - P));
            pts_imu.push_back(Q.inverse() * (C + sign.scale * G / G.norm() - P));
            pts_imu.push_back(Q.inverse() * (C + sign.scale * (N.cross(G)) / ((N.cross(G)).norm()) - P));

            for (int i = 0; i < pts_imu.size(); i++)
            {
                pts_camera.push_back(qic.inverse() * (pts_imu[i] - tic));
            }
            if (pts_camera[1].y() < pts_camera[3].y())
            {
                // change1 = true;
                Eigen::Vector3d temp;
                temp = pts_camera[1];
                pts_camera[1] = pts_camera[3];
                pts_camera[3] = temp;
            }
            if (pts_camera[2].x() < pts_camera[4].x())
            {
                // change2 = true;
                Eigen::Vector3d temp;
                temp = pts_camera[2];
                pts_camera[2] = pts_camera[4];
                pts_camera[4] = temp;
            }

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0)
                P += delta;
            else if (a == 1)
                Q = Q * Utility::deltaQ(delta);
            else if (a == 2)
                tic += delta;
            else if (a == 3)
                qic = qic * Utility::deltaQ(delta);
            else if (a == 4)
                C += delta;
            else if (a == 5)
                N += delta;
            // N = N * Utility::deltaQ(delta);
            // else if (a == 6)
            //     inv_dep_i += delta.x();
            Eigen::Matrix<double, 10, 1> tmp_residual;

            for (int i = 0; i < pts_imu.size(); i++)
            {
                tmp_residual.block<2, 1>(2 * i, 0) = (pts_camera[i] / pts_camera[i].z()).head<2>() - sign.cvPoints[i];
            }

            tmp_residual = sqrt_info * tmp_residual;
            num_jacobian.col(k) = (tmp_residual - residual) / eps;
            std::cout << tmp_residual.transpose() << std::endl;
            // num_jacobian.col(k) = (tmp_residual - residual);
        }
    }
    // std::cout << num_jacobian << std::endl;
}

// rectangle
NEWRSIGNFactor::NEWRSIGNFactor(const double p[7], const double p_[7], const SIGN sign_)
{
    pose[7] = p[7];
    ex_pose[7] = p_[7];
    sign = sign_;
};

bool NEWRSIGNFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc; // 耗时

    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]); // 载体body
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]); // 外参
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]); // 标志位置
    // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep[9];

    std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
    pts_imu.push_back(Q.inverse() * (sign.C - P));
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale * G / G.norm() - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P)); // 右上
    pts_imu.push_back(Q.inverse() * (sign.C - 0.5 * sign.scale * G / G.norm() - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P)); //  右下
    pts_imu.push_back(Q.inverse() * (sign.C - 0.5 * sign.scale * G / G.norm() + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P)); //  左下
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale * G / G.norm() + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P)); //  左上
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale * G / G.norm() - P));                                                                     // 上
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P));                                   // 左
    pts_imu.push_back(Q.inverse() * (sign.C - 0.5 * sign.scale * G / G.norm() - P));                                                                     // 下
    pts_imu.push_back(Q.inverse() * (sign.C - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P));                                   // 右

    Eigen::Map<Eigen::Matrix<double, 18, 1>> residual(residuals);
    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera.push_back(qic.inverse() * (pts_imu[i] - tic));
        inv_dep[i] = pts_camera[i].z();
        residual.block<1, 1>(2 * i, 0) << (pts_camera[i] / inv_dep[i]).x() - sign.cvPoints[i].x();
        residual.block<1, 1>(2 * i + 1, 0) << ((pts_camera[i] / inv_dep[i]).y() - sign.cvPoints[i].y()) * 0.2;
        // cout << "ProPoints:" << (pts_camera[i] / inv_dep[i]).x() << "," << (pts_camera[i] / inv_dep[i]).y() << endl;
        // cout << "cvPoints:" << sign.cvPoints[i] << endl;
    }
    residual = sqrt_rinfo * residual;
    // cout << "原始矩形残差:" << residual.transpose() << endl;
    // cout << "位置P:" << P.transpose();
    // printf("time:%f\n", sign.time);

    if (jacobians)
    {
        Eigen::Matrix3d R = Q.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 18, 27> reduce(18, 27);
        reduce.setZero();
        for (int i = 0; i < pts_camera.size(); i++)
        {
            reduce.block<2, 3>(2 * i, 3 * i) << 1.0 / inv_dep[i], 0, -pts_camera[i](0) / (inv_dep[i] * inv_dep[i]),
                0, 1.0 / inv_dep[i], -pts_camera[i](1) / (inv_dep[i] * inv_dep[i]);
        }
        // reduce << 1. / inv_dep, 0, -pts(0) / (inv_dep * inv_dep),
        //     0, 1. / inv_dep, -pts(1) / (inv_dep * inv_dep);

        reduce = sqrt_rinfo * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double, 27, 6> jaco;
            // 中
            jaco.block<3, 3>(0, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - P))); // 对Q求导
            jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[0]); // 对Q求导
            // 右上
            jaco.block<3, 3>(3, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + 0.5 * sign.scale * G / G.norm() - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[1]); // 对Q求导
            // 右下
            jaco.block<3, 3>(6, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - 0.5 * sign.scale * G / G.norm() - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[2]); // 对Q求导
            // 左下
            jaco.block<3, 3>(9, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(9, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - 0.5 * sign.scale * G / G.norm() + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(9, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[3]); // 对Q求导
            // 左上
            jaco.block<3, 3>(12, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + 0.5 * sign.scale * G / G.norm() + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[4]); // 对Q求导
            // 上
            jaco.block<3, 3>(15, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(15, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + 0.5 * sign.scale * G / G.norm() - P))); // 对Q求导
            jaco.block<3, 3>(15, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[5]); // 对Q求导
            // 左
            jaco.block<3, 3>(18, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(18, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(18, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[6]); // 对Q求导
            // 下
            jaco.block<3, 3>(21, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(21, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - 0.5 * sign.scale * G / G.norm() - P))); // 对Q求导
            jaco.block<3, 3>(21, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[7]); // 对Q求导
            // 右
            jaco.block<3, 3>(24, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(24, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            jaco.block<3, 3>(24, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[8]); // 对Q求导
            // jaco.leftCols<3>() = -ric.transpose() * R.transpose();
            // jaco.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(R.transpose() * (sign.C - P));

            jacobian_pose.leftCols<6>() = reduce * jaco;
            jacobian_pose.rightCols<1>().setZero();
        }
        // if (jacobians[1])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);

        // Eigen::Matrix<double, 27, 6> jaco_ex;
        // jaco_ex.leftCols<3>() = -ric.transpose();
        // jaco_ex.rightCols<3>() = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - P) - tic));

        // jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
        // jacobian_ex_pose.rightCols<1>().setZero();
        // }
        // if (jacobians[2])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 18, 6, Eigen::RowMajor>> jacobian_sign_pose(jacobians[2]);
        //     Eigen::Matrix<double, 27, 6> jaco_sign_pose;
        //     // jaco_sign_pose.leftCols<3>() = ric.transpose() * R.transpose();
        //     // jaco_sign_pose.rightCols<3>().setZero();

        //     jacobian_sign_pose = reduce * jaco_sign_pose;
        // }
    }
    sum_t += tic_toc.toc();
    return true;
}

void NEWRSIGNFactor::check(double **parameters)
{
    double *res = new double[18];
    double **jaco = new double *[3];
    jaco[0] = new double[18 * 7];
    jaco[1] = new double[18 * 7];
    jaco[2] = new double[18 * 6];

    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 18, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 18, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;

    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
    // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    // Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    // Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // Eigen::Vector3d pts_w = Q * pts_imu_i + P;
    // Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double inv_dep[9];

    std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
    pts_imu[0] = Q.inverse() * (sign.C - P);
    pts_imu[1] = Q.inverse() * (sign.C - P);
    pts_imu[2] = Q.inverse() * (sign.C - P);
    pts_imu[3] = Q.inverse() * (sign.C - P);
    pts_imu[4] = Q.inverse() * (sign.C - P);
    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera[i] = qic.inverse() * (pts_imu[i] - tic);
    }

    // Eigen::Vector3d pts_imu = Q.inverse() * (sign.C - P);
    // Eigen::Vector3d pts_camera = qic.inverse() * (pts_imu - tic);

    Eigen::Matrix<double, 18, 1> residual;
    for (int i = 0; i < pts_imu.size(); i++)
    {
        inv_dep[i] = pts_camera[i].z();
        residual.block<1, 1>(2 * i, 0) << (pts_camera[i] / inv_dep[i]).x() - sign.cvPoints[i].x();
        residual.block<1, 1>(2 * i + 1, 0) << ((pts_camera[i] / inv_dep[i]).y() - sign.cvPoints[i].y())*0.2;
    }

    residual = sqrt_rinfo * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 18, 19> num_jacobian;
    for (int k = 0; k < 19; k++)
    {
        Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
        // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

        double inv_dep_i = parameters[2][2];

        std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
        std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
        pts_imu[0] = Q.inverse() * (sign.C - P);
        pts_imu[1] = Q.inverse() * (sign.C - P);
        pts_imu[2] = Q.inverse() * (sign.C - P);
        pts_imu[3] = Q.inverse() * (sign.C - P);
        pts_imu[4] = Q.inverse() * (sign.C - P);
        pts_imu[5] = Q.inverse() * (sign.C - P);
        pts_imu[6] = Q.inverse() * (sign.C - P);
        pts_imu[7] = Q.inverse() * (sign.C - P);
        pts_imu[8] = Q.inverse() * (sign.C - P);
        for (int i = 0; i < pts_imu.size(); i++)
        {
            pts_camera[i] = qic.inverse() * (pts_imu[i] - tic);
        }

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            P += delta;
        else if (a == 1)
            Q = Q * Utility::deltaQ(delta);
        else if (a == 2)
            tic += delta;
        else if (a == 3)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 4)
            sign.C += delta;
        else if (a == 5)
            // N = N * Utility::deltaQ(delta);
            ;
        else if (a == 6)
            inv_dep_i += delta.x();

        Eigen::Matrix<double, 18, 1> tmp_residual;
        tmp_residual.block<2, 1>(0, 0) = sign.cvPoints[0] - (pts_camera[0] / pts_camera[0][2]).head<2>();
        tmp_residual.block<2, 1>(2, 0) = sign.cvPoints[1] - (pts_camera[1] / pts_camera[1][2]).head<2>();
        tmp_residual.block<2, 1>(4, 0) = sign.cvPoints[2] - (pts_camera[2] / pts_camera[2][2]).head<2>();
        tmp_residual.block<2, 1>(6, 0) = sign.cvPoints[3] - (pts_camera[3] / pts_camera[3][2]).head<2>();
        tmp_residual.block<2, 1>(8, 0) = sign.cvPoints[4] - (pts_camera[4] / pts_camera[4][2]).head<2>();
        tmp_residual.block<2, 1>(10, 0) = sign.cvPoints[5] - (pts_camera[5] / pts_camera[5][2]).head<2>();
        tmp_residual.block<2, 1>(12, 0) = sign.cvPoints[6] - (pts_camera[6] / pts_camera[6][2]).head<2>();
        tmp_residual.block<2, 1>(14, 0) = sign.cvPoints[7] - (pts_camera[7] / pts_camera[7][2]).head<2>();
        tmp_residual.block<2, 1>(16, 0) = sign.cvPoints[8] - (pts_camera[8] / pts_camera[8][2]).head<2>();

        tmp_residual = sqrt_rinfo * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}

// triangle
NEWTSIGNFactor::NEWTSIGNFactor(const double p[7], const double p_[7], const SIGN sign_)
{
    pose[7] = p[7];
    ex_pose[7] = p_[7];
    sign = sign_;
};

bool NEWTSIGNFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc; // 耗时

    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]); // 载体body
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]); // 外参
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]); // 标志位置
    // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep[5];
    std::vector<Eigen::Vector3d> pts_imu;          // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera;       // 关键点相机系下坐标
    pts_imu.push_back(Q.inverse() * (sign.C - P)); // 中左下右上
    pts_imu.push_back(Q.inverse() * (sign.C + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) + sign.scale * G / G.norm() * 0.28867513 - P));
    pts_imu.push_back(Q.inverse() * (sign.C - 0.57 * sign.scale * G / G.norm() - P));
    pts_imu.push_back(Q.inverse() * (sign.C - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) + sign.scale * G / G.norm() * 0.28867513 - P));
    pts_imu.push_back(Q.inverse() * (sign.C + 0.28867513 * sign.scale * G / G.norm() - P));
    Eigen::Map<Eigen::Matrix<double, 10, 1>> residual(residuals); // 五个点
    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera.push_back(qic.inverse() * (pts_imu[i] - tic));
    }
    for (int i = 0; i < pts_imu.size(); i++)
    {
        inv_dep[i] = pts_camera[i].z();
        residual.block<1, 1>(2 * i, 0) << (pts_camera[i] / inv_dep[i]).x() - sign.cvPoints[i].x();
        residual.block<1, 1>(2 * i + 1, 0) << ((pts_camera[i] / inv_dep[i]).y() - sign.cvPoints[i].y())*0.2;
        // cout << "ProPoints:" << (pts_camera[i] / inv_dep[i]).x() << "," << (pts_camera[i] / inv_dep[i]).y() << endl;
        // cout << "cvPoints:" << sign.cvPoints[i] << endl;
    }
    residual = sqrt_tinfo * residual;
    // cout << "三角形原始残差:" << residual.transpose() << endl;

    if (jacobians)
    {
        Eigen::Matrix3d R = Q.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 10, 15> reduce(10, 15);
        reduce.setZero();
        for (int i = 0; i < pts_camera.size(); i++)
        {
            reduce.block<2, 3>(2 * i, 3 * i) << 1.0 / inv_dep[i], 0, -pts_camera[i](0) / (inv_dep[i] * inv_dep[i]),
                0, 1.0 / inv_dep[i], -pts_camera[i](1) / (inv_dep[i] * inv_dep[i]);
        }
        reduce = sqrt_tinfo * reduce;
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            Eigen::Matrix<double, 15, 6> jaco;
            // 中
            jaco.block<3, 3>(0, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric(R.transpose() * (sign.C - P)); // 对Q求导
            // jaco.block<3, 3>(0, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[0]); // 对Q求导
            // 左
            jaco.block<3, 3>(3, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric(R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P)); // 对Q求导
            // jaco.block<3, 3>(3, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[1]); // 对Q求导
            // 下
            jaco.block<3, 3>(6, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C - 0.5 * sign.scale * (G / G.norm()) - P))); // 对Q求导
            // jaco.block<3, 3>(6, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[2]); // 对Q求导
            // 右
            jaco.block<3, 3>(9, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(9, 3) = ric.transpose() * Utility::skewSymmetric((R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P))); // 对Q求导
            // jaco.block<3, 3>(9, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[3]); // 对Q求导
            // 上
            jaco.block<3, 3>(12, 0) = ric.transpose() * -R.transpose(); // 对P求导
            // jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric(R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) - P)); // 对Q求导
            // jaco.block<3, 3>(12, 3) = ric.transpose() * Utility::skewSymmetric(pts_imu[4]); // 对Q求导

            jacobian_pose.leftCols<6>() = reduce * jaco;
            jacobian_pose.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 10, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            Eigen::Matrix<double, 15, 6> jaco_ex;
            // 中
            jaco_ex.block<3, 3>(0, 0) = -ric.transpose();                                                               // 对tic求导
            jaco_ex.block<3, 3>(0, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - P) - tic)); // 对ric求导
            // 左
            jaco_ex.block<3, 3>(3, 0) = -ric.transpose();                                                                                                                                                                              // 对tic求导
            jaco_ex.block<3, 3>(3, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) + 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P) - tic)); // 对ric求导
            // 下
            jaco_ex.block<3, 3>(6, 0) = -ric.transpose();                                                                                                   // 对tic求导
            jaco_ex.block<3, 3>(6, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C - 0.5 * sign.scale * (G / G.norm()) - P) - tic)); // 对ric求导
            // 右
            jaco_ex.block<3, 3>(9, 0) = -ric.transpose();                                                                                                                                                                              // 对tic求导
            jaco_ex.block<3, 3>(9, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) - 0.5 * sign.scale * (sign.N.cross(G)) / ((sign.N.cross(G)).norm()) - P) - tic)); // 对ric求导
            // 上
            jaco_ex.block<3, 3>(12, 0) = -ric.transpose();                                                                                                          // 对tic求导
            jaco_ex.block<3, 3>(12, 3) = Utility::skewSymmetric(ric.transpose() * (R.transpose() * (sign.C + 0.28867513 * sign.scale * (G / G.norm()) - P) - tic)); // 对ric求导

            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // if (jacobians[2])
        // {
        //     Eigen::Map<Eigen::Matrix<double, 10, 6, Eigen::RowMajor>> jacobian_sign_pose(jacobians[2]);
        //     Eigen::Matrix<double, 15, 6> jaco_sign_pose;
        //     Eigen::Matrix3d I;
        //     I.setIdentity();
        //     Eigen::Matrix3d Temp = I - ((N.cross(G)) / ((N.cross(G)).norm())) * (((N.cross(G)) / ((N.cross(G)).norm())).transpose()) / (N.cross(G)).norm();
        //     // 中
        //     jaco_sign_pose.block<3, 3>(0, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(0, 3).setZero();                         // 对N求导
        //     // 左
        //     jaco_sign_pose.block<3, 3>(3, 0) = ric.transpose() * R.transpose();                                                  // 对C求导
        //     jaco_sign_pose.block<3, 3>(3, 3) = ric.transpose() * R.transpose() * sign.scale * Temp * Utility::skewSymmetric(-G); // 对N求导
        //     // 下
        //     jaco_sign_pose.block<3, 3>(6, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(6, 3).setZero();                         // 对N求导
        //     // 右
        //     jaco_sign_pose.block<3, 3>(9, 0) = ric.transpose() * R.transpose();                                                   // 对C求导
        //     jaco_sign_pose.block<3, 3>(9, 3) = -ric.transpose() * R.transpose() * sign.scale * Temp * Utility::skewSymmetric(-G); // 对N求导
        //     // 上
        //     jaco_sign_pose.block<3, 3>(12, 0) = ric.transpose() * R.transpose(); // 对C求导
        //     jaco_sign_pose.block<3, 3>(12, 3).setZero();                         // 对N求导

        //     jacobian_sign_pose = reduce * jaco_sign_pose;
        // }
    }
    sum_t += tic_toc.toc();
    return true;
}

void NEWTSIGNFactor::check(double **parameters)
{
    double *res = new double[10];
    double **jaco = new double *[3];
    jaco[0] = new double[10 * 7];
    jaco[1] = new double[10 * 7];
    jaco[2] = new double[10 * 6];

    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 15, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;

    Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
    // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

    // Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    // Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // Eigen::Vector3d pts_w = Q * pts_imu_i + P;
    // Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    double inv_dep[5];

    std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
    std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
    pts_imu[0] = Q.inverse() * (sign.C - P);
    pts_imu[1] = Q.inverse() * (sign.C - P);
    pts_imu[2] = Q.inverse() * (sign.C - P);
    pts_imu[3] = Q.inverse() * (sign.C - P);
    pts_imu[4] = Q.inverse() * (sign.C - P);
    for (int i = 0; i < pts_imu.size(); i++)
    {
        pts_camera[i] = qic.inverse() * (pts_imu[i] - tic);
    }

    // Eigen::Vector3d pts_imu = Q.inverse() * (sign.C - P);
    // Eigen::Vector3d pts_camera = qic.inverse() * (pts_imu - tic);

    Eigen::Matrix<double, 10, 1> residual;
    for (int i = 0; i < pts_imu.size(); i++)
    {
        inv_dep[i] = pts_camera[i].z();
        residual.block<1, 1>(2 * i, 0) << (pts_camera[i] / inv_dep[i]).x() - sign.cvPoints[i].x();
        residual.block<1, 1>(2 * i + 1, 0) << (pts_camera[i] / inv_dep[i]).y() - sign.cvPoints[i].y();
    }

    residual = sqrt_tinfo * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 10, 19> num_jacobian;
    for (int k = 0; k < 19; k++)
    {
        Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        // Eigen::Vector3d C(parameters[2][0], parameters[2][1], parameters[2][2]);
        // Eigen::Vector3d N(parameters[2][3], parameters[2][4], parameters[2][5]);

        double inv_dep_i = parameters[2][2];

        std::vector<Eigen::Vector3d> pts_imu;    // 关键点imu系下坐标
        std::vector<Eigen::Vector3d> pts_camera; // 关键点相机系下坐标
        pts_imu[0] = Q.inverse() * (sign.C - P);
        pts_imu[1] = Q.inverse() * (sign.C - P);
        pts_imu[2] = Q.inverse() * (sign.C - P);
        pts_imu[3] = Q.inverse() * (sign.C - P);
        pts_imu[4] = Q.inverse() * (sign.C - P);
        for (int i = 0; i < pts_imu.size(); i++)
        {
            pts_camera[i] = qic.inverse() * (pts_imu[i] - tic);
        }

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            P += delta;
        else if (a == 1)
            Q = Q * Utility::deltaQ(delta);
        else if (a == 2)
            tic += delta;
        else if (a == 3)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 4)
            sign.C += delta;
        else if (a == 5)
            // N = N * Utility::deltaQ(delta);
            ;
        else if (a == 6)
            inv_dep_i += delta.x();

        Eigen::Matrix<double, 10, 1> tmp_residual;
        tmp_residual.block<2, 1>(0, 0) = sign.cvPoints[0] - (pts_camera[0] / pts_camera[0][2]).head<2>();
        tmp_residual.block<2, 1>(2, 0) = sign.cvPoints[1] - (pts_camera[1] / pts_camera[1][2]).head<2>();
        tmp_residual.block<2, 1>(4, 0) = sign.cvPoints[2] - (pts_camera[2] / pts_camera[2][2]).head<2>();
        tmp_residual.block<2, 1>(6, 0) = sign.cvPoints[3] - (pts_camera[3] / pts_camera[3][2]).head<2>();
        tmp_residual.block<2, 1>(8, 0) = sign.cvPoints[4] - (pts_camera[4] / pts_camera[4][2]).head<2>();

        tmp_residual = sqrt_tinfo * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}
