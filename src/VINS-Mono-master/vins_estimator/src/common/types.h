/*
 * IC-GVINS: A Robust, Real-time, INS-Centric GNSS-Visual-Inertial Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Geometry>

using namespace std;

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;

typedef struct GNSS
{
    double time;

    Vector3d blh;
    Vector3d std;

    bool isyawvalid;
    double yaw;
} GNSS;

typedef struct MARKER
{
    double time;
    int id;
    Vector3d pose;
    Quaterniond ori;

} MARKER;

typedef struct PVA
{
    double time;

    Vector3d blh;
    Vector3d vel;
    Vector3d att;
} PVA;

typedef struct IMU
{
    double time;
    double dt;

    Vector3d dtheta;
    Vector3d dvel;

    double odovel;
} IMU;

typedef struct Pose
{
    Matrix3d R;
    Vector3d t;
} Pose;

// 20230606_xjl
typedef struct SIGN
{
    double time;
    string signclass;
    // std::vector<Vector3d> xyz;
    double scale;

    // bool isyawvalid;
    Quaterniond q;

    // std::vector<Vector3d> Points;//对应点
    std::vector<Vector2d> cvPoints;
    std::vector<Vector2d> KeyPoints;
    Vector3d tic;
    Matrix3d ric;
    double dis;
    double similarty;
    Vector3d N;
    Vector3d C;
    std::vector<double> features;

} SIGN;

typedef struct POLE
{
    double time;

    std::deque<Vector3d> xyz;
    double scale;

    bool isyawvalid;
    Quaterniond q;
    std::deque<Vector3d> Points;
    Vector3d tic;
    Matrix3d ric;
    double dis;

} POLE;
#endif // TYPES_H
