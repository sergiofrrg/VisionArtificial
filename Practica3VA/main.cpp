//CAMBIAR EL KEYPOINTS Y EL DIRCENTRO NO ESTÁN BIEN.

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

using namespace std;

int main()
{


   cv::Mat_<uchar> image;
   cv::Mat_<uchar> image2;
   cv::ORB orb (100,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31);
   cv::Mat descriptoresImagen;
   cv::Mat conjuntoDescriptores;
   vector<std::vector<cv::KeyPoint> > keyPoints;
   vector<std::vector<cv::Point> >dirCentro;
   vector<cv::KeyPoint> kp;
   //vector<cv::Point> vectorPuntos;
   string ruta;
   //stringstream ss;
   cv::Point centro;

   cv::Mat conjuntoKeyPoints;

   for(int i=1; i<=48; i++){
       //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
       stringstream ss;
       //Dirección de labSergio
       //ss << "/home/sferrer/Documentos/VisionArtificial/Practica3/training_frontal/frontal_" << i << ".jpg";

       //Dirección de labAza
       ss << "/home/azaescudero/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << i << ".jpg";

       ruta = ss.str();
       image=cv::imread(ruta,0);

       //HALLAMOS Y GUARDAMOS LOS KEYPOINTS DE LA IMAGEN EN EL VECTOR kp Y LOS DESCRIPTORES EN LA MATRIZ descriptoresImagen
       orb.detect(image,kp);
       orb.compute(image,kp,descriptoresImagen);

       //GUARDAMOS EL CENTRO DE LA IMAGEN
       centro=cv::Point(image.cols/2, image.rows/2);

       vector<cv::Point> vectorPuntos;

       //ALMACENAMOS LA DIRECCIÓN AL CENTRO DE CADA KEYPOINT DE LA IMAGEN en vectorPuntos (y este a su vez en dirCentro)
       for (int j=0; j<kp.size(); j++){
           vectorPuntos.push_back(cv::Point(centro.x-kp.at(j).pt.x,centro.y-kp.at(j).pt.y));
       }
       dirCentro.push_back(vectorPuntos);

       //ALMACENAMOS LA MATRIZ DE DESCRIPTORES DE LA IMAGEN i EN LA MATRIZ conjuntoDescriptores
       conjuntoDescriptores.push_back(descriptoresImagen);

       //ALMAACENAMOS EL VECTOR DE KEYPOINTS DE LA IMAGEN i EN LA MATRIZ conjuntoKeyPoints
       conjuntoKeyPoints.push_back(kp);

       //keyPoints.push_back(kp);
       //cv::drawKeypoints(image, kp, image2 );
       //cv::imshow("Practica",image2);
       //cv::waitKey(0);
   }


   //CREAMOS EL ÍNDICE i PARA conjuntoDescriptores
   cv::flann::Index i (conjuntoDescriptores, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);

   cv::Mat_<uchar> image3;
   //image3=cv::imread("/home/sergiofrrg/Escritorio/aerial.png");
   //image3=cv::imread("/home/sferrer/Documentos/VisionArtificial/Practica3/Test/test1.jpg",0);

   image3=cv::imread("/home/azaescudero/Documentos/Universidad/VisionArtificial/EnunciadoP3/TestCars/Test/test1.jpg",0);
   cv::imshow("image3", image3);
   cv::waitKey();

   //HALLAMOS LOS KEYPOINTS Y DESCRIPTORES DE LA IMAGEN DE TEST
   orb.detect(image3, kp);
   orb.compute(image3,kp,descriptoresImagen);

   //BUSCAMOS LOS k VECINOS MÁS CERCANOS A LOS DESCRIPTORES DE LA IMAGEN DE TEST EN descriptoresImagen
   int k = 1;
   cv::Mat indices;
   cv::Mat dist;
   i.knnSearch(descriptoresImagen, indices, dist, k);


   //CREAMOS MATRIZ DE VOTACIÓN DEL TAMAÑO DE LA IMAGEN/bajaRes Y LO LLENAMOS DE 0s
   int bajaRes = 10;
   int matVotacion[image3.cols/bajaRes][image3.rows/bajaRes];
   for (int i = 0; i<image3.cols/bajaRes; i++){
       for (int j = 0; j<image3.rows/bajaRes; j++){
           matVotacion[i][j] = 0;
       }
   }


   //cout << "aqui"<<endl;
   //cout << indices.rows << endl;


   for (int fila=0; fila</*indices.rows*/ 40; fila++){
       //ss << endl << indices.row(fila) << endl ;
       vector<cv::KeyPoint> vecAux = keyPoints.at(indices.col(0).at(fila));
       vector<cv::Point> dirAlCentroAux = dirCentro.at(indices.col(0).at(fila));
       cout << "aqui" << fila << endl;

       for (int pos = 0; pos < vecAux.size(); pos++){
           cv::Point alCentroAux = dirAlCentroAux.at(pos);
           float escalaAprendizaje = vecAux.at(pos).size;
           float escalaTest = kp.at(0).size;
           if (escalaAprendizaje != escalaTest){
               alCentroAux.x = alCentroAux.x * (escalaTest/escalaAprendizaje);
               alCentroAux.y = alCentroAux.y * (escalaTest/escalaAprendizaje);
           }
           cv::Point centroCoche = vecAux.at(pos).pt;
           centroCoche.x += alCentroAux.x;
           centroCoche.y += alCentroAux.y;
           cout << centroCoche.x/bajaRes << " " << centroCoche.y/bajaRes << endl;
           matVotacion[centroCoche.x/bajaRes][centroCoche.y/bajaRes] ++;
       }
   }

   cv::Point centroFinal;
   int valorMayor=0;

   for (int i = 0; i<image3.cols/bajaRes; i++){
       for (int j = 0; j<image3.rows/bajaRes; j++){
           cout << matVotacion[i][j]<<endl;
           if (matVotacion[i][j]>valorMayor){
               valorMayor = matVotacion[i][j];
               centroFinal.x = j*bajaRes;
               centroFinal.y = i*bajaRes;
           }
       }
   }

   cout << centroFinal << endl;
   cout << image3.rows << " " << image3.cols << endl;
   //cv::imshow("image3", image3);
   //cv::waitKey();

   //cout << ss.str();


}
