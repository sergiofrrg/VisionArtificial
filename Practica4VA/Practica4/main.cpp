#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <vector>
#include <list>
#include <string.h>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

int main()
{
   string ruta;

   //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
   stringstream ss;
   //Dirección de labSergio
   //ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << "1" << ".jpg";
   //Dirección de labAza
   ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << 15 << ".jpg";
   ruta = ss.str();

       //Our color image
       cv::Mat imageMat = cv::imread(ruta, CV_LOAD_IMAGE_COLOR);

       //Grayscale matrix
       cv::Mat grayscaleMat (imageMat.size(), CV_8U);

       //Convert BGR to Gray
       cv::cvtColor( imageMat, grayscaleMat, CV_BGR2GRAY );

       //Binary image
       cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

       //Apply thresholding
       cv::threshold(grayscaleMat, binaryMat, 128, 255, cv::THRESH_BINARY);

       //Show the results
       cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
       cv::imshow("Output", binaryMat);

       //cv::waitKey(0);

       cv::Mat canny_output;
        vector<vector<Point> > contours;
        vector<cv::Vec4i> hierarchy;

        /// Detect edges using canny
        Canny(binaryMat, canny_output, 100, 255*2, 3 );
        /// Find contours
        findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Draw contours
        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
           {
             Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
             drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
           }

        /// Show in a window

        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );

        cv::waitKey(0);

}
