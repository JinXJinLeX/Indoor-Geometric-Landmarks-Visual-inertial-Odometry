#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits.h>
#include <float.h>
#include <iostream>

// #include "include/lapacke_config.h"  //lapack手动，未成功
// #include "include/lapacke.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0
/** PI */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif /* !M_PI */
#define M_1_2_PI 1.57079632679489661923
#define M_1_4_PI 0.785398163

#define M_3_4_PI 2.35619449

#define M_1_8_PI 0.392699081
#define M_3_8_PI 1.178097245
#define M_5_8_PI 1.963495408
#define M_7_8_PI 2.748893572
#define M_4_9_PI 1.396263401595464		  // 80°
#define M_1_9_PI 0.34906585				  // 20°
#define M_1_10_PI 0.314159265358979323846 // 18°
#define M_1_12_PI 0.261799387			  // 15°
#define M_1_15_PI 0.20943951			  // 12°
#define M_1_18_PI 0.174532925			  // 10°
/** 3/2 pi */
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI 6.28318530718
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

#define TABSIZE 100000

//=================================================================================================================
//===========================================LSD=========================================================
/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402 // ln10
#endif								  /* !M_LN10 */

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED 1

// 对于构成圆弧的像素标记极性，如果梯度的方向和弧的方向指向一致，则为SAME_POLE,否则为OPP_POLE,该标记初始是为0
#define NOTDEF_POL 0
#define SAME_POL 1
#define OPP_POL -1
/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
	int x, y;
	struct coorlist *next;
};
typedef struct ntuple_list_s
{
	int size;
	int max_size;
	int dim;
	double *values;
} *ntuple_list;
//=================================================================================================================

double *mylsd(int *n_out, double *img, int X, int Y, int **reg_img, int *reg_x, int *reg_y);
double *LineSegmentDetection(int *n_out, double *img, int X, int Y, double scale, double sigma_scale, double quant, double ang_th, double log_eps, double density_th, int n_bins, int **reg_img, int *reg_x, int *reg_y);
double calcu_dis(double x, double y, cv::Mat K, double width, double height, double ori);
int ellipse_detect(Mat input_image, int color_flag, vector<vector<double>> &output, double similarity, double xmin, double ymin);
// void calcu_p(double x,double y,cv::Mat K,double width,double height,double ori,point3d &p);
// void calcu_p(double x,double y,cv::Mat K,double calculate_dis,point3d &p);

struct point2i //(or pixel).
{
	int x, y;
};

struct point2d
{
	double x, y;
};

struct point1d1i
{
	double data;
	int cnt;
};

struct point3d
{
	double x, y;
	double r;
};
// 20221102xjl
void calcu_p(double x, double y, cv::Mat K, double calculate_dis, point3d &p);

struct point3i
{
	int x, y;
	int z;
};

struct point2d1i
{
	double x, y;
	int z;
};

struct point5d
{
	double x, y;
	double a, b;
	double phi;
};

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
	double x1, y1, x2, y2; /* first and second point of the line segment */
	double width;		   /* rectangle width */
	double x, y;		   /* center of the rectangle */
	double theta;		   /* angle */
	double dx, dy;		   /* (dx,dy) is vector oriented as the line segment,dx = cos(theta), dy = sin(theta) */
	int polarity;		   /* if the arc direction is the same as the edge direction, polarity = 1, else if opposite ,polarity = -1.*/
	double prec;		   /* tolerance angle */
	double p;			   /* probability of a point with angle within 'prec' */
};

typedef struct
{
	double vx[4];  /* rectangle's corner X coordinates in circular order */
	double vy[4];  /* rectangle's corner Y coordinates in circular order */
	double ys, ye; /* start and end Y values of current 'column' */
	int x, y;	   /* coordinates of currently explored pixel */
} rect_iter;

typedef struct image_double_s
{
	double *data;
	int xsize, ysize;
} *image_double;

typedef struct image_char_s
{
	unsigned char *data;
	unsigned int xsize, ysize;
} *image_char;

//================================Generate Ellipse Candidates=========================================
// 匹配组对，组对的索引参数，椭圆参数
typedef struct PairGroup_s
{
	point2i pairGroupInd;
	point2d center; //(x0,y0)
	point2d axis;	//(a,b)
	double phi;		// angle of orientation
} PairGroup;

// 匹配组对节点
typedef struct PairGroupNode_s
{
	point2i pairGroupInd;
	point2d center; //(x0,y0)
	point2d axis;	//(a,b)
	double phi;		// angle of orientation
	PairGroupNode_s *next;
} PairGroupNode;

typedef struct PairGroupList_s
{
	int length;
	PairGroup *pairGroup;
} PairGroupList;

typedef struct Point2dNode_s
{
	point2d point;
	Point2dNode_s *next;
} Point2dNode;

typedef struct Point3dNode_s
{
	point3d point;
	Point3dNode_s *next;
} Point3dNode;

typedef struct Point5dNode_s
{
	point2d center;
	point2d axis;
	double phi;
	Point5dNode_s *next;
} Point5dNode;

typedef struct Point1dNode_s
{
	double data;
	Point1dNode_s *next;
} Point1dNode;

/*----------------------------------------------------------------------------*/
/** int image data type

	The pixel value at (x,y) is accessed by:

	  image->data[ x + y * image->xsize ]

	with x and y integer.
 */
typedef struct image_int_s
{
	int *data;
	unsigned int xsize, ysize;
} *image_int;

