#pragma once
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
#include "../common/types.h"

/**
 * @brief This function will transform a point, given in world coordinates into image coordinates
 *
 * @param pose the six dimensional pose of the camera (rotation in rodriguez angles)
 * @param ex_pose the cameras extrinsic parameters
 * @param x_image x value of ouput point in image coordinates
 * @param y_image y value of ouput point in image coordinates
 */
template <typename T>
static inline bool transformWorldToImg(const Vector3d p_world, const T *pose, const double *pose_, const T *ex_pose, T *x_image, T *y_image)
{

    // Transform point to imu coordinates
    // T P_Imu[3];

    // world -> imu  pose[3,4,5,6] are the rotation.
    // P_Imu[0] = p_world[0] - pose[0];
    // P_Imu[1] = p_world[1] - pose[1];
    // P_Imu[2] = p_world[2] - pose[2];
    // std::cout << "auto pro:" << P_Imu[0] << "," << P_Imu[1] << "," << P_Imu[2] << std::endl;
    // T Q_Imu[4];
    // Q_Imu[0] = T(pose_[6]);
    // Q_Imu[1] = T(-pose_[3]);
    // Q_Imu[2] = T(-pose_[4]);
    // Q_Imu[3] = T(-pose_[5]);
    // T imu_result[3];
    // ceres::QuaternionRotatePoint(Q_Imu, P_Imu, imu_result);
    // Transform point to camera coordinates
    // T P_Cam[3];
    // imu -> cam  ex_pose[3,4,5,6] are the rotation.
    // P_Cam[0] = imu_result[0] - ex_pose[0];
    // P_Cam[1] = imu_result[1] - ex_pose[1];
    // P_Cam[2] = imu_result[2] - ex_pose[2];
    // T Q_Cam[4];
    // Q_Cam[0] = ex_pose[6];
    // Q_Cam[1] = -ex_pose[3];
    // Q_Cam[2] = -ex_pose[4];
    // Q_Cam[3] = -ex_pose[5];
    // T cam_result[3];
    // ceres::QuaternionRotatePoint(Q_Cam, P_Cam, cam_result);

    Eigen::Matrix<T, 3, 1> P_Imu{T(p_world[0]) - pose[0], T(p_world[1]) - pose[1], T(p_world[2]) - pose[2]};
    // Eigen::Quaternion<T> Q_Imu{T(pose_[6]), T(-pose_[3]), T(-pose_[4]), T(-pose_[5])};
    Eigen::Quaternion<T> Q_Imu{pose[6], -pose[3], -pose[4], -pose[5]};
    Eigen::Matrix<T, 3, 1> imu_result = Q_Imu * P_Imu;
    Eigen::Matrix<T, 3, 1> P_Cam{imu_result[0] - ex_pose[0], imu_result[1] - ex_pose[1], imu_result[2] - ex_pose[2]};
    Eigen::Quaternion<T> Q_Cam{ex_pose[6], -ex_pose[3], -ex_pose[4], -ex_pose[5]};
    Eigen::Matrix<T, 3, 1> cam_result = Q_Cam * P_Cam;

    if (cam_result[2] < T(0.2))
    {
        // std::cout << "WARNING; Attempt to divide by 0!" << std::endl;
        return false;
    }
    *x_image = cam_result[0] / cam_result[2];
    *y_image = cam_result[1] / cam_result[2];
    return true;
}

class SIGNFactor
{
public:
    SIGNFactor(const SIGN sign_)
    {

        // for (int i = 0; i < 7; i++)
        // {
        //     cpose[i] = p[i];
        //     cex_pose[i] = p_[i];
        // }
        sign = sign_;
        Vector3d gg = G / G.norm();
        Vector3d ff = (sign.N.cross(G)) / ((sign.N.cross(G)).norm());
        P[0] = sign.C;                         // 中
        P[1] = sign.C - 0.5 * gg * sign.scale; // 下
        P[2] = sign.C - 0.5 * ff * sign.scale; // 右
        P[3] = sign.C + 0.5 * gg * sign.scale; // 上
        P[4] = sign.C + 0.5 * ff * sign.scale; // 左
        P[5] = sign.C - 0.5 * ff * sign.scale + 0.5 * gg * sign.scale;
        P[6] = sign.C - 0.5 * ff * sign.scale - 0.5 * gg * sign.scale;
        P[7] = sign.C + 0.5 * ff * sign.scale - 0.5 * gg * sign.scale;
        P[8] = sign.C + 0.5 * ff * sign.scale + 0.5 * gg * sign.scale;
    };

    template <typename T>
    bool operator()(const T *const pose, const T *const ex_pose, T *residuals) const
    {
        T u_rmarker[9], v_rmarker[9];
        for (int i = 0; i < 9; i++)
        {
            if (transformWorldToImg(P[i], pose, cpose, ex_pose, &u_rmarker[i], &v_rmarker[i]))
            {
                // printf("circle cvPoints:%f , %f\n", sign.cvPoints[i].x(), sign.cvPoints[i].y());
                // cout << u_rmarker[i] << "," << v_rmarker[i] << endl;
                // printf("marker:%f , %f\n  ", u_rmarker[i], v_rmarker[i]);
                if (i == 0)
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                else
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                // cout << "circle" << residuals[2 * i] << endl;
                // cout << "circle" << residuals[2 * i + 1] << endl;
                // cout << "=====================================" << endl;
            }
            else
                return false;
        }
        return true;
    }

private:
    double cpose[7];
    double cex_pose[7];
    SIGN sign;
    int index;

    static Eigen::MatrixXd sqrt_info;
    static double sum_t;
    Vector3d P[9];
};

class SIGNTFactor
{
public:
    SIGNTFactor(const SIGN sign_)
    {
        // for (int i = 0; i < 7; i++)
        // {
        //     tpose[i] = p[i];
        //     tex_pose[i] = p_[i];
        // }
        sign = sign_;
        Vector3d gg = G / G.norm();
        Vector3d ff = (sign.N.cross(G)) / ((sign.N.cross(G)).norm());
        P[0] = sign.C;                                                        // 中
        P[4] = sign.C + 0.5 * ff * sign.scale + 0.28867513 * gg * sign.scale; // 左
        P[1] = sign.C - 0.57 * gg * sign.scale;                               // 下
        P[2] = sign.C - 0.5 * ff * sign.scale + 0.28867513 * gg * sign.scale; // 右
        P[3] = sign.C + 0.28867513 * gg * sign.scale;                         // 上
    };

    template <typename T>
    bool operator()(const T *const pose, const T *const ex_pose, T *residuals) const
    {

        T u_rmarker[5], v_rmarker[5];
        for (int i = 0; i < 5; i++)
        {
            if (transformWorldToImg(P[i], pose, tpose, ex_pose, &u_rmarker[i], &v_rmarker[i]))
            {
                // printf("triangle cvPoints:%f , %f\n", sign.cvPoints[i].x(), sign.cvPoints[i].y());
                // cout << u_rmarker[i] << "," << v_rmarker[i] << endl;
                // printf("marker:%f , %f\n  ", u_rmarker[i], v_rmarker[i]);
                if (i == 0)
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                else
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                // cout << "tri" << residuals[2 * i] << endl;
                // cout << "tri" << residuals[2 * i + 1] << endl;
                // cout << "=====================================" << endl;
            }
            else
                return false;
        }
        return true;
    }

private:
    double tpose[7];
    double tex_pose[7];
    SIGN sign;
    int index;
    static Eigen::MatrixXd sqrt_info;
    static double sum_t;
    Eigen::Vector3d P[5];
};

class SIGNRFactor
{
public:
    SIGNRFactor(const SIGN sign_)
    {
        // for (int i = 0; i < 7; i++)
        // {
        //     rpose[i] = p[i];
        //     rex_pose[i] = p_[i];
        // }
        sign = sign_;
        Vector3d gg = G / G.norm();
        Vector3d ff = (sign.N.cross(G)) / ((sign.N.cross(G)).norm());
        P[0] = sign.C;                                                 // 中
        P[1] = sign.C + 0.5 * gg * sign.scale - 0.5 * ff * sign.scale; // 右上
        P[2] = sign.C - 0.5 * gg * sign.scale - 0.5 * ff * sign.scale; // 右下
        P[3] = sign.C - 0.5 * gg * sign.scale + 0.5 * ff * sign.scale; // 左下
        P[4] = sign.C + 0.5 * gg * sign.scale + 0.5 * ff * sign.scale; // 左上
        P[5] = sign.C + 0.5 * gg * sign.scale;                         // 上
        P[6] = sign.C + 0.5 * ff * sign.scale;                         // 左
        P[7] = sign.C - 0.5 * gg * sign.scale;                         // 下
        P[8] = sign.C - 0.5 * ff * sign.scale;                         // 右
    };

    template <typename T>
    bool operator()(const T *const pose, const T *const ex_pose, T *residuals) const
    {
        T u_rmarker[9], v_rmarker[9];
        for (int i = 0; i < 9; i++)
        {
            if (transformWorldToImg(P[i], pose, rpose, ex_pose, &u_rmarker[i], &v_rmarker[i])) // 这个函数的输出结果没问题
            {
                // printf("rectangle cvPoints:%f , %f\n", sign.cvPoints[i].x(), sign.cvPoints[i].y());
                // cout << u_rmarker[i] << "," << v_rmarker[i] << endl;
                if (i == 0)
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                else
                {
                    residuals[2 * i] = (u_rmarker[i] - T(sign.cvPoints[i].x())) * T(460.0 * 0.2);
                    residuals[2 * i + 1] = (v_rmarker[i] - T(sign.cvPoints[i].y())) * T(460.0 * 0.2);
                }
                // cout << "rect" << residuals[2 * i] << endl;
                // cout << "rect" << residuals[2 * i + 1] << endl;
                // cout << "=====================================" << endl;
            }
            else
                return false;
        }
        // cout << "pose:" << pose[0] << "," << pose[1] << "," << pose[2] << endl;
        return true;
    }

private:
    double rpose[7];
    double rex_pose[7];
    SIGN sign;
    int index;
    static Eigen::MatrixXd sqrt_info;
    static double sum_t;
    Vector3d P[9];
};

// class MultisignFactor: public ceres::SizedCostFunction<7, 7>
class MultisignFactor
{
private:
    double pose[7];
    double Epose[7];
    // Eigen::Vector3d p1, p2;
    // Quaterniond q1, q2;
public:
    MultisignFactor(const double p[7], const double ps[7])
    {
        for (int i = 0; i < 7; i++)
        {
            pose[i] = p[i];
            // cout << "p:" << p[i] << endl;
            Epose[i] = ps[i];
            // cout << "ep:" << ps[i] << endl;
        }
    };

    template <typename T>
    bool operator()(const T *const p, T *residual) const
    {
        T relative_q[4];
        relative_q[0] = p[6];
        relative_q[1] = p[3];
        relative_q[2] = p[4];
        relative_q[3] = p[5];

        T q1[4];
        q1[0] = T(Epose[6]);
        q1[1] = T(Epose[3]);
        q1[2] = T(Epose[4]);
        q1[3] = T(Epose[5]);

        T relative_q_inv[4];
        //  relative_q 最新输入的数据 观测值相对旋转 relative_q求逆
        relative_q_inv[0] = relative_q[0];
        relative_q_inv[1] = -relative_q[1];
        relative_q_inv[2] = -relative_q[2];
        relative_q_inv[3] = -relative_q[3];

        T error_q[4];
        ceres::QuaternionProduct(relative_q, q1, error_q);
        residual[0] = (T(Epose[0]) - p[0]) / T(0.25);
        residual[1] = (T(Epose[1]) - p[1]) / T(0.25);
        residual[2] = (T(Epose[2]) - p[2]) / T(0.45);

        residual[3] = error_q[1] / T(0.02);
        residual[4] = error_q[2] / T(0.02);
        residual[5] = error_q[3] / T(0.02);
        residual[6] = error_q[4] / T(0.02);
        return true;
    }
};