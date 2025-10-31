#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();
	~PoseGraph();
	void registerPub(ros::NodeHandle &n);
	void addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);
	void loadKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);
	void loadVocabulary(std::string voc_path);
	void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1> &_loop_info);

	/**************************/
	// 20240917 xjl
	void updateKeyFrameSignLoop(double stamp, Eigen::Matrix<double, 8, 1> &_sign_loop_info);
	Vector3d sign_t_drift;
	double sign_yaw_drift;
	Matrix3d sign_r_drift;
	/**************************/

	KeyFrame *getKeyFrame(int index);
	KeyFrame *SigngetKeyFrame(double time_stamp);
	nav_msgs::Path path[10];
	nav_msgs::Path base_path;
	CameraPoseVisualization *posegraph_visualization;
	void savePoseGraph();
	void loadPoseGraph();
	void publish();
	Vector3d t_drift;
	double yaw_drift;
	Matrix3d r_drift;
	// world frame( base sequence or first sequence)<----> cur sequence frame
	Vector3d w_t_vio;
	Matrix3d w_r_vio;

private:
	int detectLoop(KeyFrame *keyframe, int frame_index);
	void addKeyFrameIntoVoc(KeyFrame *keyframe);
	void optimize4DoF();
	void updatePath();
	list<KeyFrame *> keyframelist;
	std::mutex m_keyframelist;
	std::mutex m_optimize_buf;
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;
	std::queue<int> optimize_buf;
	int global_index;
	int sequence_cnt;
	vector<bool> sequence_loop;
	map<int, cv::Mat> image_pool;
	int earliest_loop_index;
	int base_sequence;

	// 20240918 xjl
	std::thread sign_optimization;
	// void optimizeSign4DoF();
	std::queue<int> sign_optimize_buf;
	std::mutex m_sign_optimize_buf;
	int earliest_sign_loop_index;
	std::mutex m_sign_drift;
	// double yaw_sign_drift;
	// Matrix3d r_sign_drift;
	// Vector3d t_sign_drift;

	BriefDatabase db;
	BriefVocabulary *voc;

	ros::Publisher pub_pg_path;
	ros::Publisher pub_base_path;
	ros::Publisher pub_pose_graph;
	ros::Publisher pub_path[10];
};

template <typename T>
T NormalizeAngle(const T &angle_degrees)
{
	if (angle_degrees > T(180.0))
		return angle_degrees - T(360.0);
	else if (angle_degrees < T(-180.0))
		return angle_degrees + T(360.0);
	else
		return angle_degrees;
};

class AngleLocalParameterization
{
public:
	template <typename T>
	bool operator()(const T *theta_radians, const T *delta_theta_radians,
					T *theta_radians_plus_delta) const
	{
		*theta_radians_plus_delta =
			NormalizeAngle(*theta_radians + *delta_theta_radians);

		return true;
	}

	static ceres::LocalParameterization *Create()
	{
		return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
														 1, 1>);
	}
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);

	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {}

	template <typename T>
	bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i)
	{
		return (new ceres::AutoDiffCostFunction<
				FourDOFError, 4, 1, 3, 1, 3>(
			new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
};

struct FourDOFWeightError
{
	// 输入到因子图里的元素有：当前帧和回环的相对T，yaw，以及回环帧的pitch roll
	FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i)
	{
		weight = 1;
	}
	// 优化的状态有回环的那一帧，当前到回环的每一帧，换作sign的用法就是修正前30帧
	template <typename T>
	bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const
	{
		// 每一帧对回环的那帧的位移T
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9]; // 回环旧帧的ypr2R
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9]; // 求逆得imu到世界系下的R
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3]; // 求位移T在回环旧帧坐标系下的位移
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);
		// 残差为每一帧在回环旧帧坐标系下的位移和相对位移的差
		residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
		residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
		residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
		// 残差为每一帧的yaw减去回环的yaw再减去相对yaw
		residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

		return true;
	}

	static ceres::CostFunction *Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i)
	{
		return (new ceres::AutoDiffCostFunction<
				FourDOFWeightError, 4, 1, 3, 1, 3>(
			new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
	double weight;
};

struct FourDOFWeightError_Sign
{
	// 输入到因子图里的元素有：当前帧和回环的相对T，yaw，以及回环帧的pitch roll
	FourDOFWeightError_Sign(double t_x, double t_y, double t_z, double x, double y, double z, double relative_yaw, double yaw, double roll, double pitch)
		: t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), yaw(yaw), roll(roll), pitch(pitch)
	{
		weight = 1;
		relative_t.x() = t_x;
		relative_t.y() = t_y;
		relative_t.z() = t_z;
		t.x() = x + t_x;
		t.y() = y + t_y;
		t.z() = z + t_z;
		// yaw = yaw;
		// relative_yaw = relative_yaw;
	}
	// 优化的状态有回环的那一帧，当前到回环的每一帧，换作sign的用法就是修正前30帧
	template <typename T>
	bool operator()( const T *t_, T *residuals) const
	{
		// euler to rotation
		// T w_R_i[9]; // 回环旧帧的ypr2R
		// YawPitchRollToRotationMatrix(euler[0], T(pitch), T(roll), w_R_i);
		// rotation transpose
		// T i_R_w[9]; // 求逆得imu到世界系下的R
		// RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		// T t_i_ij[3]; // 求位移T在回环旧帧坐标系下的位移
		// RotationMatrixRotatePoint(i_R_w, ti, t_i_ij);
		// 残差为每一帧在回环旧帧坐标系下的位移和相对位移的差
		residuals[0] = (t_[0] - T(t.x())) * T(weight)/ T(0.01);
		residuals[1] = (t_[1] - T(t.y())) * T(weight)/ T(0.01);
		residuals[2] = (t_[2] - T(t.z())) * T(weight)/ T(0.01);
		// 残差为每一帧的yaw减去回环的yaw再减去相对yaw
		// residuals[3] = NormalizeAngle((euler[0] - T(relative_yaw + yaw))) * T(weight) / T(10.0);
		cout << "t_:" << t_[0] << "," << t_[1] << "," << t_[2] << endl;
		cout << "t:" << t[0] << "," << t[1] << "," << t[2] << endl;
		cout << "relative_t:" << t_x << "," << t_y << "," << t_z << endl;
		return true;
	}

	static ceres::CostFunction *Create(double t_x, double t_y, double t_z, double x, double y, double z,
									   double relative_yaw, double yaw, double roll, double pitch)
	{
		return (new ceres::AutoDiffCostFunction<
				FourDOFWeightError_Sign, 3, 3>(
			new FourDOFWeightError_Sign(t_x, t_y, t_z, x, y, z, relative_yaw, yaw, roll, pitch)));
	}
	double t_x, t_y, t_z;
	double relative_yaw, yaw, pitch, roll;
	Vector3d relative_t;
	Vector3d t;
	double weight;
};