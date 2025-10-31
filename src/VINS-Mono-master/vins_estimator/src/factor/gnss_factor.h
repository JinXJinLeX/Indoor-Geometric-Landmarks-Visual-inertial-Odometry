#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
#include "../common/types.h"

struct GNSSFactor
{
    double pose[7];
    GNSS gnss;

    GNSSFactor(double p[7], GNSS gnss_)
    {
        for (int i = 0; i < 7; i++)
        {
            pose[i] = p[i];
        }
        Quaterniond q = Quaterniond(pose[6], pose[3], pose[4], pose[5]);
        gnss = gnss_;
        gnss.blh = gnss_.blh - q.toRotationMatrix().transpose() * ANTLEVER;
    };

    template <typename T>
    bool operator()(const T *pose, T *residual) const
    {
        // for (int i = 0; i < 3; i++)
        // {
        //     residual[i] = (gnss.blh[i] - pose[i]) / gnss.std[i];
        // }
        residual[0] = (gnss.blh[0] - pose[0]) / gnss.std[0];
        residual[1] = (gnss.blh[1] - pose[1]) / gnss.std[1];
        residual[2] = (gnss.blh[2] - pose[2]) / gnss.std[2];
        return true;
    }
};

// class GNSSFactor : public ceres::SizedCostFunction<3, 7> {

// public:
//     explicit GNSSFactor(GNSS gnss, Vector3d lever)
//         : gnss_(std::move(gnss))
//         , lever_(std::move(lever)) {
//     }

//     void updateGnssState(const GNSS &gnss) {
//         gnss_ = gnss;
//     }

//     bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
//         Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};
//         Quaterniond q{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

//         Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

//         error = p + q.toRotationMatrix() * lever_ - gnss_.blh;

//         Matrix3d sqrt_info_ = Matrix3d::Zero();
//         sqrt_info_(0, 0)    = 1.0 / gnss_.std[0];
//         sqrt_info_(1, 1)    = 1.0 / gnss_.std[1];
//         sqrt_info_(2, 2)    = 1.0 / gnss_.std[2];

//         error = sqrt_info_ * error;

// if (jacobians) {
//     if (jacobians[0]) {
//         Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
//         jacobian_pose.setZero();

//         jacobian_pose.block<3, 3>(0, 0) = Matrix3d::Identity();
//         jacobian_pose.block<3, 3>(0, 3) = -q.toRotationMatrix() * Rotation::skewSymmetric(lever_);

//         jacobian_pose = sqrt_info_ * jacobian_pose;
//     }
// }

//         return true;
//     }

// private:
//     GNSS gnss_;
//     Vector3d lever_;
// };

// class GNSSFactor : public ceres::SizedCostFunction<3, 7>{
// public:
//   virtual ~GNSSFactor() {}
//     GNSSFactor() = delete;
//     // GNSSFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
//     // {
//     // }
//     // template <typedef T> bool operator()(const T*x,T* residual) const{
//     //     residual[0] = cost_funciton;
//     //     return true;
//     // }
//     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
//     {
//         ;
//     }
// };