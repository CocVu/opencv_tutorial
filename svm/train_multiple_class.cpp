#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <time.h>
using namespace cv;
using namespace cv::ml;
using namespace std;

// #define name_class[6] = {"neg", "motorbike", "bicycle", "car_s","car_l","walker" };
// #define data_path = "/home/nam/data/data_khanh_4k5/single_img_resize/"
// #define name_class[] = {"neg", "motorbike","walker"};
vector< float > get_svm_detector( const Ptr< SVM >& svm );
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
void test_trained_detector( String obj_det_filename, String test_dir, String videofilename );
vector< float > get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}
/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;
    const int size_x = box.width;
    const int size_y = box.height;
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols >= box.width && full_neg_lst[i].rows >= box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - size_x );
            box.y = rand() % ( full_neg_lst[i].rows - size_y );
            Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        }
    }
}
void test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );
    vector< String > files;
    //files contain name of images
    glob( test_dir, files );
    int delay = 0;
    VideoCapture cap;
    if ( videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }
    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );
    for( size_t i=0;; i++ )
    {
        Mat img;
        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = imread( files[i] );
        }
        if ( img.empty() )
        {
            return;
        }
        vector< Rect > detections;
        vector< double > foundWeights;
        // CascadeClassifier::detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())¶
        hog.detectMultiScale( img, detections, foundWeights );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
            cout << detections[j].x << endl;
            cout << detections[j].width << endl;
            cout << detections[j].y << endl;
            cout << detections[j].height << endl;
        }
        imshow( obj_det_filename, img );
        if( waitKey( delay ) == 27 )
        {
            return;
        }
    }
}
int main( int argc, char** argv )
{
  string data_path = "/home/nam/data/data_khanh_4k5/single_img_resize/";
  string name_classes[] = { "neg", "motorbike", "walker"};
  int n_classes = 3;
  String obj_det_filename = "3_class_64x128.xml";

  Size image_size = Size(64,128);

  vector< Mat > full_neg_lst, neg_lst, gradient_lst;
  vector< int > labels;

  clog << "load NEG";
  load_images( data_path + name_classes[0], full_neg_lst, false );
  clog << "load NEG";
  sample_neg( full_neg_lst, neg_lst, image_size);
  clog << "load NEG";
  int total_neg = neg_lst.size();

  clog << "HOG NEG";
  computeHOGs( image_size, neg_lst, gradient_lst, false );
  labels.assign( total_neg, -1 );

  for(int i = 1; i < n_classes; i++){
    vector <Mat> pos_lst;
    clog << "load "+ name_classes[i];
    load_images( data_path + name_classes[i], pos_lst, false );

    int total_each_class = pos_lst.size();

    clog << "HOG "+ name_classes[i];
    computeHOGs( image_size, pos_lst, gradient_lst, false );
    labels.insert( labels.end(), total_each_class, i );
  }

  Mat train_data;
  convert_to_ml( gradient_lst, train_data );


  clog << "Training SVM...";
  Ptr< SVM > svm = SVM::create();
  /* Default values to train SVM */
  svm->setCoef0( 0.0 );
  svm->setDegree( 3 );
  svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
  svm->setGamma( 0 );
  svm->setKernel( SVM::LINEAR );
  svm->setNu( 0.5 );
  svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
  svm->setC( 0.01 ); // From paper, soft classifier
  svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
  svm->train( train_data, ROW_SAMPLE, labels );
  clog << "...[done]" << endl;

  HOGDescriptor hog;
  hog.winSize = image_size;
  hog.setSVMDetector( get_svm_detector( svm ) );
  hog.save( obj_det_filename );
  // test_trained_detector( obj_det_filename, test_dir, videofilename );
  return 0;
}
