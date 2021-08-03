/*
 Utils function.
 Here there are function not in the algorithm, but usefull.
 Created by Paolo Forin on 18/07/21.
*/

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>

void load_image(std::string pos_dir, std::string neg_dir, std::vector<cv::Mat> &images, std::vector<int> &id_classes, cv::Size size_image, std::string end_path_pos, std::string end_path_neg);
void convert2ml(std::vector<cv::Mat> &data, cv::Mat &train);
float compute_iou(cv::Rect rectA, cv::Rect rectB);
void file_to_vectorOfLines(std::string path_file, std::vector<std::string> &imagesName, std::vector< std::vector<cv::Rect> > &boxesImage);
void create_samples(std::string path_file, std::string path_dir_dataset, std::string boat_save_path, std::string no_boat_save_path, int neg_width, int neg_height, int n_not_boat);

#endif /* utils_hpp */
