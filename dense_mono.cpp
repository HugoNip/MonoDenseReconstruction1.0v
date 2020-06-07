#include "common.h"

std::string file_dir = "../data";

int main() {

    // read data
    std::vector<std::string> color_image_files;
    std::vector<Sophus::SE3d> poses_TWC;
    cv::Mat ref_depth;
    bool ret = readDatasetFiles(file_dir, color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        std::cout << "reading image files failed" << std::endl;
        return -1;
    }
    std::cout << "Read total " << color_image_files.size() << " files." << std::endl;

    // reference image
    cv::Mat ref = cv::imread(color_image_files[0], 0); // gray-scale image
    Sophus::SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0; // initial depth mu
    double init_cov2 = 3.0; // initial depth sigma^2
    cv::Mat depth(height, width, CV_64F, init_depth); // depth map
    cv::Mat depth_cov2(height, width, CV_64F, init_cov2); // depth sigma^2 map

    for (int index = 1; index < color_image_files.size(); ++index) {
        std::cout << "*** loop " << index << " ***" << std::endl;
        cv::Mat curr = cv::imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        Sophus::SE3d pose_curr_TWC = poses_TWC[index];
        Sophus::SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // pose transform: T_C_W * T_W_R  = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        cv::imshow("image", curr);
        cv::waitKey(1);
    }

    std::cout << "estimation returns,saving depth map ..." << std::endl;
    cv::imwrite("../results/depth.png", depth);
    std::cout << "done." << std::endl;

    return 0;
}
