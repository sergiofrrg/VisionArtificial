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
#include <infokeypoint.h>

using namespace std;

int main()
{
   cv::Mat_<uchar> image;
   cv::ORB orb (10,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31);
   cv::Mat_<uchar> descriptoresImagen;
   cv::Mat_< cv::Mat_<uchar> > conjuntoDescriptores;
   vector<std::vector<cv::KeyPoint> > keyPoints;
   vector<std::vector<cv::Point> >dirCentro;
   vector<cv::KeyPoint> kp;
   //vector<cv::Point> vectorPuntos;
   string ruta;
   cv::Point centro;

   //cv::Mat conjuntoKeyPoints;

   vector< vector<InfoKeyPoint> > conjuntoInfoKeyPoints;
   vector<InfoKeyPoint> listaInfoKeyPoints;

   for(int i=1; i<=48; i++){
       //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
       stringstream ss;
       //Dirección de labSergio
       //ss << "/home/sferrer/Documentos/VisionArtificial/Practica3/training_frontal/frontal_" << i << ".jpg";

       //Dirección de labAza
       ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << i << ".jpg";

       ruta = ss.str();
       image=cv::imread(ruta,0);

       //HALLAMOS Y GUARDAMOS LOS KEYPOINTS DE LA IMAGEN EN EL VECTOR kp Y LOS DESCRIPTORES EN LA MATRIZ descriptoresImagen
       orb.detect(image,kp);
       orb.compute(image,kp,descriptoresImagen);

       //GUARDAMOS EL CENTRO DE LA IMAGEN
       centro=cv::Point(image.cols/2, image.rows/2);

       //vector<cv::Point> vectorPuntos;

       std::vector<InfoKeyPoint> vectorInfoKP; //si esto en vez de un vector pones que sea cv::Mat, dice que memory corruption

       //ALMACENAMOS LA DIRECCIÓN AL CENTRO DE CADA KEYPOINT DE LA IMAGEN Y EL PROPIO KEYPOINT
       //en vectorInfoKP (y este a su vez en conjuntoInfoKeyPoints)
       for (int j=0; j<kp.size(); j++){
           //vectorPuntos.push_back(cv::Point(centro.x-kp.at(j).pt.x,centro.y-kp.at(j).pt.y));
           vectorInfoKP.push_back(InfoKeyPoint(kp.at(j),
                                               cv::Point(centro.x-kp.at(j).pt.x,
                                                         centro.y-kp.at(j).pt.y)));
          listaInfoKeyPoints.push_back(InfoKeyPoint(kp.at(j),
                                               cv::Point(centro.x-kp.at(j).pt.x,
                                                         centro.y-kp.at(j).pt.y)));

           /*cout << "imagen: " << i << ", tamaño: " << image.cols << "x" << image.rows << ", "
                   << "kp número: " << j << ", punto: " << vectorInfoKP.at(j).getKeyPoint().pt << endl; */
       }

       //dirCentro.push_back(vectorPuntos);

       conjuntoInfoKeyPoints.push_back(vectorInfoKP);

       //ALMACENAMOS LA MATRIZ DE DESCRIPTORES DE LA IMAGEN i EN LA MATRIZ conjuntoDescriptores
       conjuntoDescriptores.push_back(descriptoresImagen);

       /* Esto ya no nos sirve --------------
       //ALMACENAMOS EL VECTOR DE KEYPOINTS DE LA IMAGEN i EN LA MATRIZ conjuntoKeyPoints
       conjuntoKeyPoints.push_back(kp);

       //keyPoints.push_back(kp);
       //cv::drawKeypoints(image, kp, image2 );
       //cv::imshow("Practica",image2);
       //cv::waitKey(0);
       --------------------------------------*/

   }


   //CREAMOS EL ÍNDICE i PARA conjuntoDescriptores
   cv::flann::Index i (conjuntoDescriptores, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);

   cv::Mat_<uchar> image3;
   //image3=cv::imread("/home/sergiofrrg/Escritorio/aerial.png");
   //image3=cv::imread("/home/sferrer/Documentos/VisionArtificial/Practica3/Test/test1.jpg",0);

   image3=cv::imread("/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/TestCars/Test/test1.jpg",0);

   //cv::imshow("image3", image3);
   //cv::waitKey();

   //HALLAMOS LOS KEYPOINTS Y DESCRIPTORES DE LA IMAGEN DE TEST
   orb.detect(image3, kp);
   orb.compute(image3,kp,descriptoresImagen);

   //BUSCAMOS LOS k VECINOS MÁS CERCANOS A LOS DESCRIPTORES DE LA IMAGEN DE TEST EN descriptoresImagen
   int k = 3;
   cv::Mat_<int> indices;
   cv::Mat dist;
   i.knnSearch(descriptoresImagen, indices, dist, k);

   //cout << "index: " << endl << i. << endl;
   cout << "indices: " << endl << indices << endl;
   cout << "número de keypoints imagen test: " << endl << kp.size() << endl;
   cout << "descriptores Imagen Test: "<< endl << descriptoresImagen << endl;
   cout << "tamaño descriptores img test: " << descriptoresImagen.rows << "x" << descriptoresImagen.cols << endl;
   cout << "tamaño matriz descriptores aprendizaje: " << conjuntoDescriptores.rows << "x" << conjuntoDescriptores.cols << endl;
   cout << "tamaño vector de vectores infoKeyPoints: " << conjuntoInfoKeyPoints.size() << endl;
   cout << "tamaño lista infoKeyPoints: " << listaInfoKeyPoints.size();

   //CREAMOS MATRIZ DE VOTACIÓN DEL TAMAÑO DE LA IMAGEN/bajaRes Y LO LLENAMOS DE 0s
   int bajaRes = 10;
   int matVotacion[image3.cols/bajaRes][image3.rows/bajaRes];
   for (int i = 0; i<image3.cols/bajaRes; i++){
       //cout << i << " - ";
       for (int j = 0; j<image3.rows/bajaRes; j++){
           matVotacion[i][j] = 0;
           //cout << matVotacion[i][j];
       }
       //cout << endl;
   }

   int escalaImagenTest = kp.at(0).size;
   //cout << escalaImagenTest;


   //OBTENEMOS LOS INFOKEYPOINTS DE APRENDIZAJE CORRESPONDIENTES A LOS DESCRIPTORES DE
   //APRENDIZAJE MÁS PARECIDOS A LOS DE LA IMAGEN TEST

   //Segundo intento
   /*for (int k = 0; k<indices.cols; k++){
       for (int poskp = 0; poskp<indices.col(k).rows; poskp++){
           //Guardamos el keypoint de la imagen aprendizaje k y su dircentro
           InfoKeyPoint infoKpAux = conjuntoInfoKeyPoints.row(indices.row(poskp).row(k));
           cv::KeyPoint kpAux = infoKpAux.getKeyPoint();
           cv::Point dirCentroAux = infoKpAux.getVectorCentro();

           //Modificamos la escala de donde supuestamente está el centro comparando la de la
           //imagen test con la de la imagen de aprendizaje (de cualquiera de sus keypoints)
           int escalaAprendizajeImagenK = kpAux.size;
           dirCentroAux.x = dirCentroAux.x * (escalaImagenTest/escalaAprendizajeImagenK);
           dirCentroAux.y = dirCentroAux.y * (escalaImagenTest/escalaAprendizajeImagenK);

           //Le sumamos el dirCentroAux al keyPoint actual de la imagen test para obtener el
           //supuesto centro
           cv::Point votoCentro;
           votoCentro.x = kp.at(poskp).pt.x + dirCentroAux.x;
           votoCentro.y = kp.at(poskp).pt.y + dirCentroAux.y;

           //Votamos dividiendo las coordenadas entre bajaRes
           matVotacion[votoCentro.x/bajaRes][votoCentro.y/bajaRes] ++;
       }
   }
   */

   //Tercer intento
/*
   for (cv::Mat_<int>::iterator it = indices.begin(); it!=indices.end(); it++){
       InfoKeyPoint infoKpAux = conjuntoInfoKeyPoints.at(*it);
   }

   int valorMayor = 0;
   cv::Point centroFinal;

   for (int i = 0; i<image3.cols/bajaRes; i++){
       for (int j = 0; j<image3.rows/bajaRes; j++){
           cout << matVotacion[i][j]<<endl;
           if (matVotacion[i][j]>valorMayor){
               valorMayor = matVotacion[i][j];
               centroFinal.x = j*bajaRes;
               centroFinal.y = i*bajaRes;
           }
       }
   }*/

   //Primer intento
   /* for (int fila=0; fila<indices.rows; fila++){
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

   //DESPUÉS DE LA VOTACIÓN, OBTENEMOS EL CENTRO

           //Le sumamos el dirCentroAux al keyPoint actual de la imagen test para obtener el
           //supuesto centro
           cv::Point votoCentro;
           votoCentro.x = kp.at(poskp).pt.x + dirCentroAux.x;
           votoCentro.y = kp.at(poskp).pt.y + dirCentroAux.y;

           //Votamos dividiendo las coordenadas entre bajaRes
           matVotacion[votoCentro.x/bajaRes][votoCentro.y/bajaRes] ++;

   cout << centroFinal << endl;
   cout << image3.rows << " " << image3.cols << endl;
   //cv::imshow("image3", image3);
   //cv::waitKey();

   //cout << ss.str();

*/
}
