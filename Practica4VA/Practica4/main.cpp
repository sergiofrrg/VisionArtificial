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
#include <subspace.hpp>

using namespace std;
using namespace cv;
using namespace libfacerec;

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
        findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );

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

        for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); ++it)
        {
            cout << contourArea(*it)                                                                                                                                                                                << endl;
        }

        cout << contours.size() << endl;

        //Parte 2 de la práctica
        cv::Mat_<uchar> digito;
        string clases [37] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ESP", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
        vector<string> e;
        vector<int> eNteros;
        cv::Mat_<int> mCaracteristicas;
        //Cargamos los '1' como prueba

        int nClase = 0;

        for(int i=1; i<=9250; i++){

            stringstream ss;
            //Dirección de labSergio

            int aux = i % 250;
            if (aux == 0)
                aux = 250;

                        //cout << aux << " " << nClase << endl;

            ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP4/Digitos/" << clases[nClase] << "_" << aux << ".jpg";

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
            cv::Mat_<int> fila;

            cv::MatIterator_<uchar> it;
            for( it = binaryDigit.begin<uchar>(); it != binaryDigit.end<uchar>(); ++it){

                if((int)*it==255){
                    fila.push_back(1);
                }else{
                    fila.push_back(0);
                }
                ++j;

            }

            e.push_back(clases[nClase]);
            eNteros.push_back(nClase);
            mCaracteristicas.push_back((cv::Mat_<int>)fila.t());

            if (i % 250 == 0)
                nClase++;

        }

        LDA lda(mCaracteristicas, eNteros);
        cv::Mat_<int> cr;
        cr = lda.project(mCaracteristicas);

        cout << mCaracteristicas.rows;
        cout << cr.row(0);


//        cout << mCaracteristicas.row(0) << endl;
//        for(int j=0; j<100; j++){
//                        cout << vCaracteristicas[0][j] << ", ";
//                    }
//                    cout << endl;


//        for (int i=0; i<250; i++){
//            for(int j=0; j<100; j++){
//                cout << vCaracteristicas[i][j] << " ";
//            }
//            cout << endl;
//        }

//        for (int i = 0; i<e.size(); i++){
//            cout << e.at(i) << endl;
//        }


}
