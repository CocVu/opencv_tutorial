#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define CLASSES 5

#define MOTORBIKE 0
#define BICYCLE 1
#define PERSON 2
#define CAR_S 3
#define CAR_L 4

#define NUM_FEATURES 2

using namespace cv;
using namespace cv::ml;

int n_point = 5;

float * blob_extract_feature(int Blob){

}

int main(int, char**)
{
    // Set up training data
    int labels[4] = {MOTORBIKE, BICYCLE, PERSON, CAR_L};
    float trainingData[4][NUM_FEATURES] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, NUM_FEATURES, CV_32F, trainingData);
    Mat labelsMat(4, 1, CV_32SC1, labels);

    // Train the SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    svm->save("abc.xml");

    // // Data for visual representation
    // int width = 512, height = 512;
    // Mat image = Mat::zeros(height, width, CV_8UC3);
    // // Show the decision regions given by the SVM
    // for (int i = 0; i < image.rows; i++)
    // {
    //     for (int j = 0; j < image.cols; j++)
    //     {
    //         Mat sampleMat = (Mat_<float>(1,2) << j,i);
    //         float response = svm->predict(sampleMat);
    //         // draw diff color for each classes
    //         image.at<Vec3b>(i,j)  = Vec3b(255/CLASSES * response,255/CLASSES * response,255 - (255/CLASSES * response) );
    //     }
    // }

    // // Show the training data
    // int thickness = -1;
    // circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness );
    // circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness );
    // circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness );
    // circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness );

    // // Show support vectors
    // thickness = 2;
    // Mat sv = svm->getUncompressedSupportVectors();
    // for (int i = 0; i < sv.rows; i++)
    // {
    //     const float* v = sv.ptr<float>(i);
    //     circle(image,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness);
    // }
    // imwrite("result.png", image);        // save the image
    // imshow("SVM Simple Example", image); // show it to the user
    // waitKey();
    return 0;
}
