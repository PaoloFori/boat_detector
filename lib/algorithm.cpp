/*
 Implement the algorithm.
 I used HOG features with Selective Search Segmentation and SVM.
 Created by Paolo Forin on 18/07/21.
*/

#include "algorithm.hpp"
#include <iostream>
#include <fstream>
#include "utils.hpp" //for IoU, convert2ml, pre_processing and check_rect
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> //for imread
#include <opencv2/ximgproc/segmentation.hpp> // for selective search
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp> //for svm and multimap
#include <opencv2/objdetect.hpp> //for hog
#include <opencv2/core/utils/filesystem.hpp> // for glob

void Boat_detector::test(cv::String path_svm, cv::String test_dir, cv::String type_test, cv::Size size, cv::String path_txt_test, bool show_iou){
    
    //load model
    std::cout << "[INFO] load model...";
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    try{
        svm = cv::ml::SVM::load(path_svm);
        std::cout << "...[done]" << std::endl;
    }catch(std::exception e){
        std::cout << "[ERROR] load model, see if the path is correct" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    //have images paths
    std::vector<cv::String> files; //contains always the path of images
    cv::utils::fs::glob(test_dir, type_test, files); //to have path images if show_iou is false
    std::vector< std::vector<cv::Rect> > ground_truth; //inizialization we don't use it if show_iou is false
    
    if(show_iou){
        std::cout << "[INFO] extracting ground of truth of test images...";
        std::vector< std::string > files_name;
        file_to_vectorOfLines(path_txt_test, files_name, ground_truth);
        std::cout << "...[done]" << std::endl;
        
        //path in the same order of ground of truth
        files.clear();
        for(int i = 0; i < files_name.size(); i++){
            files.push_back(test_dir + "/" + files_name[i]);
        }
    }
    
    for (size_t i = 0; i < files.size(); i++){
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::steady_clock::now();
        std::cout << "[INFO] processing image " << i+1 << " of " << files.size() << std::endl;
        
        //image usefull for the algorithm
        cv::Mat image, image_resize, image_noResize, image_noResize_edges, image_edges, image_afterNMS;
        cv::Mat image_box_found, final_image, image_gt;
        std::vector<cv::Rect> boxes_found; //will contains boxes that have score grather than a threshold
        
        //load the image
        image = cv::imread(files[i]);
        
        //calculate the edges images, usefull to discard some recrs given by ss
        pre_processing(image, image_edges);

        //clone the images, so noResize means that them are in the original size
        image_noResize = image.clone();
        image_noResize_edges = image_edges.clone();
        
        //Resize the image to improve ss
        double scale = (float) 450 / image.size().width;
        cv::resize(image, image_resize, cv::Size(0, 0), scale, scale, cv::INTER_AREA);
        cv::resize(image_edges, image_edges, cv::Size(0, 0), scale, scale, cv::INTER_AREA);
        
        //image where will draw the boxes founded
        image_box_found = image_resize.clone();

        //prepare rects returned by ss and score of probability for the algoritm
        std::vector<cv::Rect> rects;
        std::vector<float> scores; //contain the score for all rects
        
        std::cout << "  [INFO] do selective search..." << std::endl;
        cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
        ss->setBaseImage(image_resize);
        ss->switchToSelectiveSearchFast();
        ss->process(rects);
        
        //count how many rect we skip using the pre processing
        int cont_pass = 0;
        
        //calculate probability for each rect to be a boat
        for (int j = 0; j < rects.size(); j++) {
            
            //calculate back in the original image the position and the size of the rect
            cv::Rect roiRect;
            roiRect.x = (int)(rects[j].x * 1 / scale);
            roiRect.y = (int)(rects[j].y * 1 / scale);
            roiRect.width = (int)(rects[j].width * 1 / scale);
            roiRect.height = (int)(rects[j].height * 1 / scale);

            //fast check to not go out of the image with the roi
            if((roiRect.x + roiRect.width > image_noResize.cols)){
                roiRect.width = image_noResize.cols - roiRect.x;
            }
            if(roiRect.y + roiRect.height > image_noResize.rows){
                roiRect.height = image_noResize.rows - roiRect.y;
            }

            //variable usefull to evaluate the rect
            cv::Mat X_test, roi, roi_check;
            
            //check if we need to see the rect or not
            roi_check = image_noResize_edges(roiRect);
            if(!check_rect(roi_check)){
                cont_pass += 1;
                continue; //no white points so we skip to another rect
            }
            
            //extract roi and resize it to have back a correct number of features extracted with hog
            roi = image_noResize(roiRect);
            cv::resize(roi, roi, size, cv::INTER_AREA);
            
            //extract hog features and transform them in the correct way
            std::vector<cv::Mat> gradients;
            std::vector<cv::Mat> roi_passed = {roi};
            extract_hog(size, roi_passed, gradients);
            convert2ml(gradients, X_test);
            
            //predict if in the rect there are a boat
            float y_test = svm->predict(X_test, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT); //distance to the margin
            y_test = 1.0 / (1.0 + exp(-y_test)); //calculate the probability

            //if the probability is greater than 0.5 is a boat, so draw the rect
            if(y_test > 0.5){
                //remember the rect and its score
                boxes_found.push_back(rects[j]);
                scores.push_back(y_test);
                //draw rect and its score
                rectangle(image_box_found, rects[j], cv::Scalar(0, 0, 255));
                cv::putText(image_box_found, std::to_string(y_test), cv::Point(rects[j].x, rects[j].y), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));
            }
        }
        //show image with all predicted boxes
        cv::imshow("Before nms:", image_box_found);
        
        //we show the IoU only if in the input we have the path of the ground of truth
        if(show_iou){
            //we have the txt of the test, so we show also the iou
            //ground of truth for the current image
            image_gt = image_resize.clone();
            std::vector<cv::Rect> gt = ground_truth[i];
            //calculate the ground of truth of the resized image and show them
            for(size_t j = 0; j < gt.size(); j++){
                cv::Rect gt_use; //we need to resize the gt beacuse we work with a image resized
                gt_use.x = (int) (gt[j].x * scale);
                gt_use.y = (int) (gt[j].y * scale);
                gt_use.width = (int) (gt[j].width * scale);
                gt_use.height = (int) (gt[j].height * scale);
                
                //this two if allow to not exit form the image with the ground of truth
                if((gt_use.x + gt_use.width > image_noResize.cols)){
                    gt_use.width = image_noResize.cols - gt_use.x;
                }
                
                if(gt_use.y + gt_use.height > image_noResize.rows){
                    gt_use.height = image_noResize.rows - gt_use.y;
                }
                
                //draw all ground of truth of the image
                cv::rectangle(image_gt, gt_use, cv::Scalar(0,0,255));
            }
            cv::imshow("Ground of truth", image_gt);
            
            //do nms on the boxes predicted
            image_afterNMS = image_resize.clone();
            std::vector<cv::Rect> out; //boxes that remain after nms
            std::vector<float> scores_out; //scores of the boxes that remain
            //we want at least one boxes
            int neighbors = 7; //number of neightbors of a detected rect
            double minScoreSum = neighbors * 0.82; //number of score summed in nms
            while(true){
                nms(boxes_found, scores, out, scores_out, 0.03, neighbors, minScoreSum);
                if(out.size() >= gt.size()){ //at least one detected rect after applied nms
                    break;
                }else{
                    if(neighbors > 0){ //we try removing neighbors required
                        neighbors -= 1;
                        minScoreSum = neighbors * 0.82;
                    }else if(neighbors == 0){ //means no sufficient rects detected
                        break;
                    }
                }
            }
            //draw rect and their scores
            for(int i = 0; i < out.size(); i++){
                cv::rectangle(image_afterNMS, out[i], cv::Scalar(0,0,255));
                cv::putText(image_afterNMS, std::to_string(scores_out[i]), out[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));
            }
            cv::imshow("After nms", image_afterNMS);
            
            //final image
            final_image = image_resize.clone();
            for(size_t j = 0; j < out.size(); j++){
                float max_iou = 0;
                //for each rect calculate the max IoU wrt the ground of truth of the image
                for(size_t k = 0; k < gt.size(); k++){
                    //calculate back the ground of truth of the image
                    cv::Rect gt_use;
                    gt_use.x = (int) (gt[k].x * scale);
                    gt_use.y = (int) (gt[k].y * scale);
                    gt_use.width = (int) (gt[k].width * scale);
                    gt_use.height = (int) (gt[k].height * scale);
                    
                    //this two if allow to not exit form the image withe the ground of truth
                    if((gt_use.x + gt_use.width > image_noResize.cols)){
                        gt_use.width = image_noResize.cols - gt_use.x;
                    }
                    if(gt_use.y + gt_use.height > image_noResize.rows){
                        gt_use.height = image_noResize.rows - gt_use.y;
                    }
                    
                    //calculate IoU and remember the best one
                    float iou = compute_iou(out[j], gt_use);
                    if(max_iou < iou){
                        max_iou = iou;
                    }
                }
                
                //draw the rect based on the IoU
                if(max_iou >= 0.5){
                    cv::rectangle(final_image, out[j], cv::Scalar(0,255,0));
                    cv::putText(final_image, "IoU:" + std::to_string(max_iou), out[j].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,0));
                }
                else if (max_iou > 0.0){
                    cv::rectangle(final_image, out[j], cv::Scalar(0,255,255));
                    cv::putText(final_image, "IoU:" + std::to_string(max_iou), out[j].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,255));
                }
                if(max_iou <= 0.0){
                    cv::rectangle(final_image, out[j], cv::Scalar(0,0,255));
                    cv::putText(final_image, "IoU:" + std::to_string(max_iou), out[j].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,255));
                }
            }
            cv::imshow("Final results", final_image);
        }else{
            //we don't have the txt of the ground of truth, so we show only the detected
            //do nms on the boxes predicted
            image_afterNMS = image_resize.clone();
            std::vector<cv::Rect> out; //boxes that remain after nms
            std::vector<float> scores_out; //scores of the boxes that remain
            //we want at least one boxes
            int neighbors = 7; //number of neightbors of a detected rect
            double minScoreSum = neighbors * 0.82; //number of score summed in nms
            while(true){
                nms(boxes_found, scores, out, scores_out, 0.03, neighbors, minScoreSum);
                if(out.size() >= 1){ //at least one detected rect after applied nms
                    break;
                }else{
                    if(neighbors > 0){ //we try removing neighbors required
                        neighbors -= 1;
                        minScoreSum = neighbors * 0.82;
                    }else if(neighbors == 0){ //means no sufficient rects detected
                        break;
                    }
                }
            }
            //draw rect and their scores
            for(int i = 0; i < out.size(); i++){
                cv::rectangle(image_afterNMS, out[i], cv::Scalar(0,0,255));
                cv::putText(image_afterNMS, std::to_string(scores_out[i]), out[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255));
            }
            cv::imshow("After nms", image_afterNMS);
        }
        
        //print out last consideration
        std::cout << "  [INFO] rects skipped: " << cont_pass << " of " << rects.size()-1 << std::endl;
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "  [TIME] time to process image: " << elapsed_seconds.count() << "s\n" << std::endl;
        cv::waitKey(0); //show all windows
        cv::destroyAllWindows(); 
    }
}

/*
 train a svm and save it
 */
void Boat_detector::train_SVM(std::vector<cv::Mat> &gradients, std::vector<int> &y_train, std::string path_save_svm){
    
    //prepare data to train
    cv::Mat X_train;
    convert2ml(gradients, X_train);

    //start svm with default parameters
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::steady_clock::now();
    cv::Ptr< cv::ml::SVM > svm = cv::ml::SVM::create();
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3));
    svm->setGamma(0);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setNu(0.5);
    svm->setP(0.1); // for EPSILON_SVR
    svm->setC(0.01); // From paper, soft classifier
    svm->setType(cv::ml::SVM::EPS_SVR); // do regression task
    svm->train(X_train, cv::ml::ROW_SAMPLE, y_train);
    std::cout << "...[done]" << std::endl;
    //print time to train the svm
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "[TIME] time to train svm: " << elapsed_seconds.count() << "s\n" << std::endl;
    
    //save the svm
    std::cout << "[INFO] saving svm...";
    svm->save(path_save_svm);
    std::cout << "...[done]" << std::endl;
}

/*
 return true if there are at leat 25 points not black in the image passed
    we pass an image that has only point in gray scale
 */
bool Boat_detector::check_rect(cv::Mat image){
    bool white_points = false;
    int cont = 0;
    //look all pixels for the image
    for(int j = 0; j < image.rows; j++){
        for(int k = 0; k < image.cols; k++){
            cv::Vec3b pixel = image.at<cv::Vec3b>(j, k);
            if(pixel[0] > 0){ //found white point
                if(cont > 25){
                    break;
                }
                cont += 1;
            }
        }
        if(cont > 25){ //rect has 25 white points so pass it
            white_points = true;
            break;
        }
    }
    return white_points;
}

/*
 simple pre processing that is usefull to have an edges image to use for removing some not interest rects
   in selective search segmentation
*/
void Boat_detector::pre_processing(cv::Mat &src_image, cv::Mat &dst_image){
    cv::Mat image;
    image = src_image.clone();
    
    cv::Mat image1, gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY); //color gray
    cv::GaussianBlur(gray_image, image1, cv::Size(9,9), 20); //gaussian on color gray
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(0, 0));
    
    cv::Mat canny;
    dst_image = gray_image - image1; //difference to have something similar to edges
    cv::threshold(dst_image, dst_image, 10, 255, 0); //threshold to have only interesent point
    cv::GaussianBlur(gray_image, gray_image, cv::Size(5,5), 20); //gaussian again gray image
    cv::Canny(gray_image, canny, 100, 200); //canny on gray image
    dst_image = canny - dst_image; // difference from 2 way to have edges (it isn't a binary image, we can have gray points)
}

/*
 compute the hog features for all images passed and save the features on gradients
 */
void Boat_detector::extract_hog(cv::Size windowSize, std::vector<cv::Mat> &images, std::vector<cv::Mat> &gradients) {
    
    //initialize hog with the parameters
    cv::HOGDescriptor hog;
    hog.winSize = windowSize;
    hog.nbins = 9;
    hog.cellSize = cv::Size(4, 4);
    
    //variable for hog
    cv::Mat gray;
    std::vector<float> descriptors;

    for (size_t i = 0; i < images.size(); i++){
        
        //look only images that for sure don't get an error due to the dimension of windows
        if (images[i].cols >= windowSize.width && images[i].rows >= windowSize.height){
            
            //resize the image, so all images return the same number of gradients
            cv::resize(images[i], images[i], windowSize, cv::INTER_AREA);
            cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        
            //extract hog and put it in the output vector
            hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            gradients.push_back(cv::Mat(descriptors).clone()); //use .clone() if remove it not works the transpose
        }
    }
}


/*
 do the nms based on the probabilty of the rect, number of neighbors and min score sum
 the return parameter is resRect that contains the rect that remain
 */
void Boat_detector::nms(std::vector<cv::Rect> &srcRects, std::vector<float> &scores, std::vector<cv::Rect> &resRects, std::vector<float> &scores_out, float thresh, int neighbors, float minScoresSum){
    resRects.clear();

    //if no input rects
    const size_t size = srcRects.size();
    if (!size)
        return;

    //sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(scores[i], i);
    }

    // eep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        //grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];
        float rect1_score = scores[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            //grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            //compute IoU
            float overlap = compute_iou(rect1, rect2);

            //if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        //compute the rects that return back
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum){
            resRects.push_back(rect1);
            scores_out.push_back(rect1_score);
        }
    }
}

/*
 Construct
 */
Boat_detector::Boat_detector(){
}
