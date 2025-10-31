#pragma once
#include <iostream>

// #include <ros/assert.h>
// #include <ceres/ceres.h>
// #include <eigen3/Eigen/Dense>
// #include "../utility/utility.h"
// #include "../utility/tic_toc.h"
// #include "../parameters.h"
// #include "../common/types.h"

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"
#include "../common/types.h"

// ellipse
class NEWSIGNFactor : public ceres::SizedCostFunction<10, 7, 7>
{
public:
    // NEWSIGNFactor(const double p[7], const double p_[7], const double sp[6] , const std::vector<Vector2d> pts);
    NEWSIGNFactor(const double p[7], const double p_[7], const SIGN sign_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    double pose[7];
    double ex_pose[7];
    SIGN sign;
    // Eigen::Vector3d nc;
    int index;

    static Eigen::MatrixXd sqrt_info;
    static double sum_t;
};

// rectangle
class NEWRSIGNFactor : public ceres::SizedCostFunction<18, 7, 7>
{
public:
    // NEWSIGNFactor(const double p[7], const double p_[7], const double sp[6] , const std::vector<Vector2d> pts);
    NEWRSIGNFactor(const double p[7], const double p_[7], const SIGN sign_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    double pose[7];
    double ex_pose[7];
    SIGN sign;
    // Eigen::Vector3d nc;
    int index;

    static Eigen::MatrixXd sqrt_rinfo;
    static double sum_t;
};
// triangle
class NEWTSIGNFactor : public ceres::SizedCostFunction<10, 7, 7>
{
public:
    // NEWSIGNFactor(const double p[7], const double p_[7], const double sp[6] , const std::vector<Vector2d> pts);
    NEWTSIGNFactor(const double p[7], const double p_[7], const SIGN sign_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    double pose[7];
    double ex_pose[7];
    SIGN sign;
    // Eigen::Vector3d nc;
    int index;

    static Eigen::MatrixXd sqrt_tinfo;
    static double sum_t;
};
