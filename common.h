#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <fstream>
#include <boost/timer.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ------------------------------------------------------
// define parameters
const int boarder = 20; // boarder width
const int width = 640; // image
const int height = 480; // image
const double fx = 481.2f; // camera intrinsics
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3; // half-size width of NCC sample window
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // area of NCC sample window
const double min_cov = 0.1; // convergence: min sigma
const double max_cov = 10; // scattered: max sigma

// ------------------------------------------------------
// define important functions

// read data from REMODE
bool readDatasetFiles(
        const std::string &path,
        std::vector<std::string> &color_image_files,
        std::vector<Sophus::SE3d> &poses,
        cv::Mat &ref_depth);

/**
 * update depth estimation according to the new input image
 * @param ref            reference frame
 * @param curr           current frame
 * @param T_C_R          pose from Reference frame to Current frame
 * @param depth_mu       depth
 * @param depth_cov2     depth sigma^2
 * @return               successful
 */
bool update(
        const cv::Mat &ref,
        const cv::Mat &curr,
        const Sophus::SE3d &T_C_R,
        cv::Mat &depth_mu,
        cv::Mat &depth_cov2);

/**
 * epipolar search
 * @param ref                         reference frame
 * @param curr                        current frame
 * @param T_C_R                       pose from R to C
 * @param pt_ref                      point in reference frame
 * @param depth_mu                    depth average value
 * @param depth_cov2                  depth sigma^2
 * @param pt_curr                     current point
 * @param epipolar_direction          the direction of epopolar
 * @return                            successful
 */
bool epipolarSearch(
        const cv::Mat &ref,
        const cv::Mat &curr,
        const Sophus::SE3d &T_C_R,
        const Eigen::Vector2d &pt_ref,
        const double &depth_mu,
        const double &depth_cov2,
        Eigen::Vector2d &pt_curr,
        Eigen::Vector2d &epipolar_direction);

/**
 * update depth filter
 * @param pt_ref                      point in reference frame
 * @param pt_curr                     point in current frame
 * @param T_C_R                       pose from R to C
 * @param epipolar_direction          epipolar direction
 * @param depth_mu                    depth mu
 * @param depth_cov2                  depth sigma^2
 * @return                            successful
 */
bool updateDepthFilter(
        const Eigen::Vector2d &pt_ref,
        const Eigen::Vector2d &pt_curr,
        const Sophus::SE3d &T_C_R,
        const Eigen::Vector2d &epipolar_direction,
        cv::Mat &depth_mu,
        cv::Mat &depth_cov2);

/**
 * compute NCC score
 * @param ref                reference frame
 * @param curr               current frame
 * @param pt_ref             point in reference frame
 * @param pt_curr            point in current frame
 * @return                   NCC score
 */
double NCC(
        const cv::Mat &ref,
        const cv::Mat &curr,
        const Eigen::Vector2d &pt_ref,
        const Eigen::Vector2d &pt_curr);

// bilinear interpolated value
inline double getBilinearInterpolateValue(
        const cv::Mat &img,
        const Eigen::Vector2d &pt
        ) {
    uchar* d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
             xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------
// display depth map
void plotDepth(const cv::Mat &depth_truth,
        const cv::Mat &depth_estimate);

// coordinate transform from pixel to camera coordinate
inline Eigen::Vector3d px2cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d(
            (px(0, 0) - cx) / fx,
            (px(1, 0) - cy) / fy,
            1);
}

// coordinate transform from camera to pixel coordinate
inline Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(
            p_cam(0, 0) * fx / p_cam(2, 0) + cx,
            p_cam(1, 0) * fy / p_cam(2, 0) + cy);
}

// detect a point is inside the boarder
inline bool inside(const Eigen::Vector2d &pt) {
    return pt(0, 0) >= boarder &&
           pt(1, 0) >= boarder &&
           pt(0, 0) + boarder < width &&
           pt(1, 0) + boarder < height;
}

// show epipolar match
void showEpipolarMatch(const cv::Mat &ref,
        const cv::Mat &curr,
        const Eigen::Vector2d &px_ref,
        const Eigen::Vector2d &px_curr);

// show epipolar line
void showEpiploarLine(const cv::Mat &ref,
        const cv::Mat &curr,
        const Eigen::Vector2d &px_ref,
        const Eigen::Vector2d &px_min_curr,
        const Eigen::Vector2d &px_max_curr);

// evaluate depth
void evaluateDepth(const cv::Mat &depth_truth,
        const cv::Mat &depth_estimate);

#endif //COMMON_H
