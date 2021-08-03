/*
 Utils function.
 Here there are function not in the algorithm, but usefull.
 Created by Paolo Forin on 18/07/21.
*/

#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp> // for glob
#include <opencv2/ximgproc/segmentation.hpp> // for selective search

/*
 load negative and positive images, resize all of them into size_image. All image same size so after same number of features.
    Create a vector of labels, 1 if boat, -1 if not boat.
 */
void load_image(std::string pos_dir, std::string neg_dir, std::vector<cv::Mat> &images, std::vector<int> &id_classes, cv::Size size_image, std::string end_path_pos, std::string end_path_neg){
    //calculate clock for load image
    std::cout << "[INFO] start load images..." << std::endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::steady_clock::now();
    
    //***POSITIVE***
    //variable for positive image
    std::vector<cv::String> pos_paths;
    
    //load positive samples and create labels with ones -> if positive label is 1
    cv::utils::fs::glob(pos_dir, end_path_pos, pos_paths);
    for(int i = 0; i < pos_paths.size(); i++){

        //read all the image and push it into a vector
        cv::Mat image = cv::imread(pos_paths.at(i));
        if(image.empty()){
            std::cout << "[ERROR] file: "<< pos_paths.at(i) << " Not Found" << std::endl;
            exit(EXIT_FAILURE);
        }
        cv::resize(image, image, size_image); //Resize the image so after we extract the same number of features with hog
        images.push_back(image);
        id_classes.push_back(1);
    }
    std::cout << "[INFO] number of positive images: " << images.size() << std::endl;

    //***NEGATIVE***
    //variable for negative images
    std::vector<cv::Mat> neg_images;
    std::vector<int> id_neg;
    
    //to remember all path
    std::vector<cv::String> paths;
    cv::utils::fs::glob(neg_dir, end_path_neg, paths);
    
    //load only negative images
    for(int i = 0; i < paths.size(); i++){
        //read all the image and push it into a vector
        cv::Mat image = cv::imread(paths.at(i));
        if(image.empty()){
            std::cout << "[ERROR] file: "<< paths.at(i) << " Not Found" << std::endl;
            exit(EXIT_FAILURE);
        }
        //extract at random a block to same size of the positive image
        cv::resize(image, image, size_image);
        neg_images.push_back(image);
        id_classes.push_back(-1);
    }
    
    //put all togheter
    images.insert(images.end(), neg_images.begin(), neg_images.end());
    id_classes.insert(id_classes.end(), id_neg.begin(), id_neg.end());
    std::cout << "[INFO] number of negative images: " << neg_images.size() << std::endl;
    std::cout << "[INFO] total number of images: " << images.size() << std::endl;
    
    //fast check
    if(images.size() != id_classes.size()){
        std::cout << "[ERROR] error not same size into image loaded and labels used";
        exit(EXIT_FAILURE);
    }
    
    //print time to load images
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "[TIME] time to load all images: " << elapsed_seconds.count() << "s\n" << std::endl;
}

//convert a vector of Mat (those Mats are or 1xn or nx1) into only one Mat with size (size of vector data x n)
void convert2ml(std::vector<cv::Mat> &data, cv::Mat &train){
    //Convert data
    const int rows = (int)data.size();
    const int cols = (int)std::max(data[0].cols, data[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1); // used for transposition if needed
    train = cv::Mat(rows, cols, CV_32FC1);
    for (size_t i = 0; i < data.size(); ++i){
        CV_Assert(data[i].cols == 1 || data[i].rows == 1); //check that is a 'vector'
        if (data[i].cols == 1){
            transpose(data[i], tmp);
            tmp.copyTo(train.row((int)i));
        }
        else if (data[i].rows == 1){
            data[i].copyTo(train.row((int)i));
        }
    }
}

//compute IoU
float compute_iou(cv::Rect rectA, cv::Rect rectB){
    //found top left and bottom right points
    cv::Point tlA = rectA.tl();
    cv::Point tlB = rectB.tl();
    cv::Point brA = rectA.br();
    cv::Point brB = rectB.br();
    //found intersection points top left and bottom right
    int i_tl_x = std::max(tlA.x, tlB.x);
    int i_tl_y = std::max(tlA.y, tlB.y);
    int i_br_x = std::min(brA.x, brB.x);
    int i_br_y = std::min(brA.y, brB.y);
    
    //calculate area and IoU
    int inter_area = std::max(i_br_x-i_tl_x, 0) * std::max(i_br_y-i_tl_y,0);
    float iou = ((float) inter_area) / ((float) (rectA.area() + rectB.area() - inter_area));
    return iou;
}

//givne the path of teh grando of truth (.txt), it returns name and boxes of the images
void file_to_vectorOfLines(std::string path_file, std::vector<std::string> &imagesName, std::vector< std::vector<cv::Rect> > &boxesImage){

    //open stream
    std::ifstream f(path_file);
    if(f.is_open()){
        //get line and save usefull information
        std::string line;
        while(std::getline(f, line)){
            //Get name of the image
            int last_slash = (int) line.find_last_of("/");
            int first_space = (int) line.find(" ");
            std::string nameImage = line.substr(last_slash+1, first_space-last_slash-1); //with -1 we don't have the space at the end of the name
            imagesName.push_back(nameImage);
            
            //get number of boxes and parameter of boxes
            line.erase(0, first_space + 1); //in this way we have only the numerical part of string line
            int second_space = (int) line.find(" ");
            std::string numberOfBoxes = line.substr(0, second_space);
            int boxes = std::stoi(numberOfBoxes); //number of boxes for that image
            std::vector<cv::Rect> boxesParam; //it is going to contain all rects for the image
            line.erase(0, second_space + 1);
            std::string s = line; //line contains all the rect
            for(int i = 0; i < boxes; i++){ //to see all rect
                std::vector<int> params;
                for(int j = 0; j < 4; j++){ //one rect has 4 parameters
                    int space = (int) s.find_first_of(" ");
                    std::string value = s.substr(0, space);
                    params.push_back(std::stoi(value));
                    s.erase(0, space + 1);
                }
                //save parameters of one rect
                cv::Rect rect;
                rect.x = params[0];
                rect.y = params[1];
                rect.width = params[2];
                rect.height = params[3];
                boxesParam.push_back(rect);
            }
            boxesImage.push_back(boxesParam);//put all together
        }
        f.close();
    }else{
        std::cout << "[ERROR] file: " + path_file << " Not Found" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

//create the dataset for positive and negative samples, using the ground of truth and the provide images
void create_samples(std::string path_file, std::string path_dir_dataset, std::string boat_save_path, std::string no_boat_save_path, int neg_width, int neg_height, int n_not_boat){
    //take all name and the ground of truth of the images
    std::vector<std::string> imagesName;
    std::vector< std::vector<cv::Rect> > boxesImage;
    
    //return the boxes for all the image and their name. Required the path to the .txt file of ground of truth
    file_to_vectorOfLines(path_file, imagesName, boxesImage);
    
    //general count used to save the image created
    int total_count_boat = 0;
    int total_count_no_boat = 0;
    
    //see all the images
    for(size_t i = 0; i < imagesName.size(); i++){
        std::cout << "[INFO] process image: " << imagesName[i] << std::endl;
        std::cout << "  [INFO] image number " << i << " of " << imagesName.size() << std::endl;
        
        //load the image
        std::string image_path = path_dir_dataset + "/" + imagesName[i];
        cv::Mat image = cv::imread(image_path);
        cv::Mat image_copy;
        image.copyTo(image_copy); //we will extract the roi from image copy
        
        //gound of truth of the images
        std::vector<cv::Rect> grounds_truth = boxesImage[i];
        
        //count for boat or not_boat in the image
        int count_boat = 0;
        int count_no_boat = 0;
        int total_count = 0; //total count in the image, so we don't see all the sub image returned by ss
        
        //extract the ground of truth of the image and classified them as boat
        for(cv::Rect gt : grounds_truth){
            cv::Range cols(gt.x, gt.x + gt.width);
            cv::Range rows(gt.y, gt.y + gt.height);
            cv::Mat image_roi = image_copy(rows, cols);
            cv::Mat image_roi_use;
            image_roi.copyTo(image_roi_use);
            //path to save the classified boat
            std::string save_path = boat_save_path + "/boat_" + std::to_string(total_count_boat) + ".jpg";
            cv::imwrite(save_path, image_roi_use);
            count_boat +=1;
            total_count_boat +=1;
            std::cout << "  [INFO] positive image created from ground of truth: " << std::to_string(count_boat) << "/" << std::to_string(grounds_truth.size()) << std::endl;
        }
        
        //for selective search segmentation
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
        //do selective search on the image
        std::vector<cv::Rect> results;
        ss->setBaseImage(image);
        ss->switchToSelectiveSearchFast();
        ss->process(results);
        
        //look rectangles given by selective search
        for(cv::Rect foundBox : results){
            cv::Rect foundBox_use = cv::Rect(foundBox.x, foundBox.y, foundBox.width, foundBox.height);
            //we select only retangle that are enough big
            if(foundBox_use.width >= neg_width && foundBox_use.height >= neg_height){
                cv::Range cols(foundBox.x, foundBox.x + foundBox.width);
                cv::Range rows(foundBox.y, foundBox.y + foundBox.height);
                cv::Mat image_roi = image_copy(rows, cols);
                float max = 0;
                float iou;
                //compute the IoU with the ground of truth of the image (one image can have more than one boxes, for this we use the max)
                for(size_t j = 0; j < boxesImage.at(i).size(); j++){
                    iou = compute_iou(foundBox_use, boxesImage.at(i).at(j));
                    if(iou > max){
                        max = iou;
                    }
                }
                iou = max;
                //we look to a boxes with small IoU so we are sure that it is not a boat
                if(iou < 0.001){
                    // we do not want to much new element from one image
                    if(count_no_boat < n_not_boat){
                        cv::Mat image_roi_use;
                        image_roi.copyTo(image_roi_use); //to remove if we use a 128x128 samples
                        std::string save_path = no_boat_save_path + "/no_boat_" + std::to_string(total_count_no_boat) + ".jpg";
                        cv::imwrite(save_path, image_roi_use);
                        count_no_boat += 1;
                        total_count_no_boat += 1;
                        std::cout << "  [INFO] negative image created..." << std::to_string(count_no_boat) << "/" << n_not_boat << " (iou: " << std::to_string(iou) << ")" << std::endl;
                    }
                }
                //we do not want to see all ss
                if(total_count >= 1000 || (count_no_boat >= n_not_boat) ){
                    break; //we go to another image
                }
                total_count += 1;
            }
        }
    }
}
