/*
 boat detector.
 Implement an algorithm that use HOG features with Selective Search Segmentation and SVM.
 Created by Paolo Forin on 18/07/21.
*/

#include <iostream>
#include <opencv2/core/utility.hpp> // for commandLineParser
#include "utils.hpp"
#include "algorithm.hpp"

int main(int argc, const char * argv[]) {
    
    //parse the input, eg. -pos_dir=path_dir_pos or --pos_dir=...
    const char *keys = {
        "{help h         |          | show help message}"
        "{pos_dir        |          | path of directory contains positive images}"
        "{neg_dir        |          | path of directory contains negative images}"
        "{test_dir       |          | path of directory contains test images}"
        "{gt_test        |          | path of ground of truth of test images}"
        "{gt_train       |          | path of ground of truth of train images}"
        "{type_neg       |.jpg      | extension positive images}"
        "{type_pos       |.jpg      | extension negative images}"
        "{type_test      |.jpg      | extension test images}"
        "{only_test      |true      | test a trained detector with ground of truth of test images}"
        "{only_detect    |false     | test a trained detector without ground of truth of test images}"
        "{create_sample  |false     | create positive and negative samples}"
        "{train_test     |false     | tran a new model and test it (allowed value on only_detect and only_test)}"
        "{model_path     |../svm.yml| load a given model}"
    };
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        exit(EXIT_FAILURE);
    }

    cv::String pos_dir = parser.get<cv::String>("pos_dir");
    cv::String neg_dir = parser.get<cv::String>("neg_dir");
    cv::String test_dir = parser.get<cv::String>("test_dir");
    cv::String gt_test = parser.get<cv::String>("gt_test");
    cv::String gt_train = parser.get<cv::String>("gt_train");
    cv::String type_neg = "*" + parser.get<cv::String>("type_neg");
    cv::String type_pos = "*" + parser.get<cv::String>("type_pos");
    cv::String type_test = "*" + parser.get<cv::String>("type_test");
    cv::String model_path = parser.get<cv::String>("model_path");
    bool only_test = parser.get<bool>("only_test");
    bool train_test = parser.get<bool>("train_test");
    bool only_detect = parser.get<bool>("only_detect");
    bool create_sample = parser.get<bool>("create_sample");
    
    if(create_sample){
        if(gt_train.empty() || pos_dir.empty() || neg_dir.empty()){
            parser.printMessage();
            std::cout << "The following parameters are required: gt_train, pos_dir, neg_dir" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::string path_dataset, width_s, height_s, n_not_s;
        int width, height, n_not_boat;
        std::cout << "[REQUIRED] path dataset from how extract positive and negative images: ";
        std::cin >> path_dataset;
        std::cout << "[REQUIRED] min width negative samples: ";
        std::cin >> width_s;
        width = std::stoi(width_s);
        std::cout << "[REQUIRED] min height negative samples: ";
        std::cin >> height_s;
        height = std::stoi(height_s);
        std::cout << "[REQUIRED] max number of negative samples for an image: ";
        std::cin >> n_not_s;
        n_not_boat = std::stoi(n_not_s);
        create_samples(gt_train, path_dataset, pos_dir, neg_dir, width, height, n_not_boat);
        std::cout << "[INFO] created all samples" << std::endl;
        std::exit(EXIT_SUCCESS); //did what requiered
    }
    
    //create an istance of the algorithm
    Boat_detector algo = Boat_detector();
    //parameter of hog
    cv::Size size_hog = cv::Size(136, 136); //size that works well 136x136
    
    //check what required
    if(train_test){
        //chek parameters
        if(pos_dir.empty() || neg_dir.empty() || test_dir.empty()){
            parser.printMessage();
            std::cout << "[REQUIRED] the following parameters are required: pos_dir, neg_dir, test_dir" << std::endl;
            exit(EXIT_FAILURE);
        }
        //paramters usefull to prepare training data
        std::vector<cv::Mat> images; //contains all images (pos and neg)
        std::vector<int> classes; //contains all labels for the images up
        load_image(pos_dir, neg_dir, images, classes, size_hog, type_pos, type_neg);
        
        //training data
        std::vector<cv::Mat> gradients;
        algo.extract_hog(size_hog, images, gradients);
        
        //train svm and also convert the data in the correct form to train a SVM
        std::cout << "[INFO] train svm...";
        algo.train_SVM(gradients, classes, model_path);
        
    }
    
    //check what required
    if(only_detect){
        //chek parameters
        if(model_path.empty() || test_dir.empty()){
            parser.printMessage();
            std::cout << "[REQUIRED] the following parameters are required: model_path, test_dir" << std::endl;
            exit(EXIT_FAILURE);
        }
        //do what required
        gt_test = cv::String(); // do not need the gt_test, we want only to see the detected rects
        std::cout << "\n[INFO] start testing" << std::endl;
        algo.test(model_path, test_dir, type_test, size_hog, gt_test, false);
        exit(EXIT_SUCCESS); //did what required
    }
    
    //check what required
    if(only_test){
        //check what required
        if(model_path.empty() || test_dir.empty() || gt_test.empty()){
            parser.printMessage();
            std::cout << "[REQUIRED] the following parameters are required: model_path, test_dir, gt_test" << std::endl;
            exit(EXIT_FAILURE);
        }
        //do what required
        std::cout << "\n[INFO] start testing" << std::endl;
        algo.test(model_path, test_dir, type_test, size_hog, gt_test, true);
        exit(EXIT_SUCCESS); //did what required
    }
    
}
