/*
 Implement the algorithm.
 I used HOG features with Selective Search Segmentation and SVM.
 Created by Paolo Forin on 18/07/21.
*/

#ifndef algorithm_hpp
#define algorithm_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

class Boat_detector{
public:
    Boat_detector();
    void nms(std::vector<cv::Rect> &srcRects, std::vector<float> &scores, std::vector<cv::Rect> &resRects, std::vector<float> &scores_out, float thresh, int neighbors, float minScoresSum);
    void extract_hog(cv::Size windowSize, std::vector<cv::Mat> &images, std::vector<cv::Mat> &gradients);
    void pre_processing(cv::Mat &src_image, cv::Mat &dst_image);
    bool check_rect(cv::Mat image);
    void train_SVM(std::vector<cv::Mat> &X_train, std::vector<int> &y_train, std::string path_save_svm);
    void test(cv::String path_svm, cv::String test_dir, cv::String type_test, cv::Size size, cv::String path_txt_test, bool show_iou);
};

#endif /* algorithm_hpp */
