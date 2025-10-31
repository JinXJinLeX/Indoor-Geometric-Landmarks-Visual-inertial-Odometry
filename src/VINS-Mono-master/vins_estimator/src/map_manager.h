#ifndef SIGN_MANAGER_H
#define SIGN_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "common/types.h"

class SignPerFrame
{
public:
  SignPerFrame(const vector<Vector2d> pts_, const double time_)
  {
    for (int i = 0; i < pts_.size(); i++)
    {
      pts.push_back(pts_[i]);
    }
    time = time_;
  }
  SIGN sign;
  double time;
  string sign_class;    // 标志类别
  Vector3d C;           // 标志中心点世界系坐标
  Vector3d N;           // 标志法向量世界系坐标
  vector<Vector2d> pts; // 标志关键点
};

class SignPerId // 滑窗内所有标志
{
public:
  const int sign_id;
  const string classify;
  int is_detected[WINDOW_SIZE];
  double size;
  vector<SignPerFrame> sign_per_frame;

  int used_num;
  bool is_outlier;
  // bool is_margin;
  // double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
  Vector3d C_;    // 标志中心点世界系坐标
  Vector3d N_;
  double time;//最后一次观测的时间

  // Vector3d gt_p;

  SignPerId(int _sign_id, string classify, Vector3d C_, Vector3d N_, int used_num_)
      : sign_id(_sign_id), classify(classify), C_(C_), N_(N_), used_num(used_num_), solve_flag(0)
  // used_num(0), estimated_depth(-1.0), solve_flag(0)
  {
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
      is_detected[i] = 0;
    }
  }
};

class LocalMapManager
{
public:
  // LocalMapManager();
  LocalMapManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearsignState();

  void removesignBack();
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removesignFront();
  void addSignCheck(Vector3d C, string c, int &id);
  void initialSign(int &id, string c, Vector3d C, Vector3d N, double time_, vector<Vector2d> pc, int flag);
  void removesignOutlier();
  void debugShowsign();
  // void updateSign(Vector3d np, string classofsign, int id);
  void updateSign(double *time);
  int getSignCount();
  int getFeatureCount();
  // deque<SIGN> signlist;
  // deque<POLE> polelist;

  list<SignPerId> sign;

private:
  Matrix3d *Rs;
  Matrix3d ric[NUM_OF_CAM];
};

#endif