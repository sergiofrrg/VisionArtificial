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
   ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << "15" << ".jpg";
   //Dirección de labAza
   //ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << 15 << ".jpg";
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
       //cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
       //cv::imshow("Binary Image", binaryMat);               //Imagen del coche binarizada

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

        //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );      //Se dibujan los contornos encontrados
        //imshow( "Contours", drawing );

        //cv::waitKey(0);

        //Parte 2 de la práctica
        cv::Mat_<uchar> digito;
        int vCaracteristicas[9250][100];
        vector<char> e;

        //Cargamos los '1' como prueba

        for(int i=1; i<=9250; i++){

            stringstream ss;
            //Dirección de labSergio
            ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP4/Digitos/1_" << i << ".jpg";

            //Dirección de labAza
            //ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << i << ".jpg";

            ruta = ss.str();
            digito=cv::imread(ruta,0);


            //Se reescala la imagen del dígito a 10x10

            //cv::Mat_<int> resizedDigit(10, 10, DataType<int>::type);

            cv::Mat resizedDigit;

            Size size(10,10);

            //cout << "Filas: " << resizedDigit.rows << " Columnas; " << resizedDigit.cols << endl << "FIN 1" << endl;

            cv::resize(digito,resizedDigit,size,0,0,INTER_LINEAR );



            //Binarizamos los dígitos, no hace falta transformarlos a escala de gris porque ya están, da error si utilizas el cvtColor por eso.

            cv::Mat binaryDigit(resizedDigit.size(), resizedDigit.type());

            cv::threshold(resizedDigit, binaryDigit, 128, 255, cv::THRESH_BINARY);


            //Centrar los Digitos ¿?

            //Convertir las imágenes a matrices de 1x100

            int j=0;

            cv::MatIterator_<uchar> it;
            for( it = binaryDigit.begin<uchar>(); it != binaryDigit.end<uchar>(); ++it){

                if((int)*it==255)
                    vCaracteristicas[i-1][j]=1;
                else
                    vCaracteristicas[i-1][j]=0;


                ++j;

            }




        }

//        for (int i=0; i<250; i++){
//            for(int j=0; j<100; j++){
//                cout << vCaracteristicas[i][j] << " ";
//            }
//            cout << endl;
//        }




}
