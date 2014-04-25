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
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

int main()
{
   cv::Mat_<uchar> image;
   cv::ORB orb (100,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31);
   cv::Mat_<uchar> descriptoresImagen;
   cv::Mat_< cv::Mat_<uchar> > conjuntoDescriptores;
   vector<std::vector<cv::KeyPoint> > keyPoints;
   vector<std::vector<cv::Point> >dirCentro;
   vector<cv::KeyPoint> kp;
   //vector<cv::Point> vectorPuntos;
   string ruta;
   cv::Point centro;

   //cv::Mat conjuntoKeyPoints;

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

       //ALMACENAMOS LA DIRECCIÓN AL CENTRO DE CADA KEYPOINT DE LA IMAGEN Y EL PROPIO KEYPOINT
       //en vectorInfoKP (y este a su vez en conjuntoInfoKeyPoints)
       for (int j=0; j<kp.size(); j++){
          listaInfoKeyPoints.push_back(InfoKeyPoint(kp.at(j),
                                               cv::Point(centro.x-kp.at(j).pt.x,
                                                         centro.y-kp.at(j).pt.y)));
       }

       //ALMACENAMOS LA MATRIZ DE DESCRIPTORES DE LA IMAGEN i EN LA MATRIZ conjuntoDescriptores
       conjuntoDescriptores.push_back(descriptoresImagen);
   }


   //CREAMOS EL ÍNDICE i PARA conjuntoDescriptores
   cv::flann::Index i (conjuntoDescriptores, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);

   cv::Mat_<uchar> image3;
   //image3=cv::imread("/home/sergiofrrg/Escritorio/aerial.png");
   //image3=cv::imread("/home/sferrer/Documentos/VisionArtificial/Practica3/Test/test1.jpg",0);

   image3=cv::imread("/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/TestCars/Test/test1.jpg",0);

   //HALLAMOS LOS KEYPOINTS Y DESCRIPTORES DE LA IMAGEN DE TEST
   orb.detect(image3, kp);
   orb.compute(image3,kp,descriptoresImagen);

   //BUSCAMOS LOS k VECINOS MÁS CERCANOS A LOS DESCRIPTORES DE LA IMAGEN DE TEST EN descriptoresImagen
   int k = 10;
   cv::Mat_<int> indices;
   cv::Mat dist;
   i.knnSearch(descriptoresImagen, indices, dist, k);

   cout << "indices: " << endl << indices << endl;
   cout << "número de keypoints imagen test: " << endl << kp.size() << endl;
   cout << "descriptores Imagen Test: "<< endl << descriptoresImagen << endl;
   cout << "tamaño descriptores img test: " << descriptoresImagen.rows << "x" << descriptoresImagen.cols << endl;
   cout << "tamaño matriz descriptores aprendizaje: " << conjuntoDescriptores.rows << "x" << conjuntoDescriptores.cols << endl;
   cout << "tamaño lista infoKeyPoints: " << listaInfoKeyPoints.size();

   //CREAMOS MATRIZ DE VOTACIÓN DEL TAMAÑO DE LA IMAGEN/bajaRes Y LO LLENAMOS DE 0s
   int bajaRes = 10;
   int matVotacion[image3.cols/bajaRes][image3.rows/bajaRes];
   for (int i = 0; i<image3.cols/bajaRes; i++){
       for (int j = 0; j<image3.rows/bajaRes; j++){
           matVotacion[i][j] = 0;
       }
   }

   int escalaImagenTest = kp.at(0).size;

   //OBTENEMOS LOS INFOKEYPOINTS DE APRENDIZAJE CORRESPONDIENTES A LOS DESCRIPTORES DE
   //APRENDIZAJE MÁS PARECIDOS A LOS DE LA IMAGEN TEST

   int contadorKP = 0;
   int contadorColumna = 1;

   for (cv::Mat_<int>::iterator it = indices.begin(); it!=indices.end(); it++){
       if (contadorColumna > k){
           contadorKP ++;
           contadorColumna = 1;
       }

       //Guardamos el keypoint de la imagen aprendizaje k y su dircentro
       InfoKeyPoint infoKpAux = listaInfoKeyPoints.at(*it);
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
       votoCentro.x = kp.at(contadorKP).pt.x + dirCentroAux.x;
       votoCentro.y = kp.at(contadorKP).pt.y + dirCentroAux.y;

       cout << "iteración: " << contadorKP << " Indice: " << *it << " Voto al punto: " << votoCentro << endl;

       //Votamos dividiendo las coordenadas entre bajaRes si no busca en un punto fuera de la imagen
       if ((votoCentro.x <= image3.cols) && (votoCentro.y <= image3.rows))
            matVotacion[votoCentro.x/bajaRes][votoCentro.y/bajaRes] ++;

       contadorColumna ++;
   }


   //DESPUÉS DE LA VOTACIÓN, BUSCAMOS EL PUNTO QUE TIENE MÁS VOTOS

   int valorMayor = 0;
   cv::Point centroFinal;

   cout << "matriz votación final: " << endl;
   for (int i = 0; i<image3.cols/bajaRes; i++){
       for (int j = 0; j<image3.rows/bajaRes; j++){
           cout << matVotacion[i][j] << " ";
           if (matVotacion[i][j]>valorMayor){
               valorMayor = matVotacion[i][j];
               centroFinal.x = j*bajaRes;
               centroFinal.y = i*bajaRes;
           }
       }
       cout << endl;
   }
   cout << centroFinal << endl;

   //AQUI COMIENZA LO DEL HAAR

   void detectAndDisplay( cv::Mat frame );

   cv::String car_cascade_name = "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/haar/coches.xml";
   cv::CascadeClassifier car_cascade;
   string window_name = "Car Detection";
   cv::RNG rng(12345);

   CvCapture* capture;
   cv::Mat frame;

   //-- 1. Load the cascades
   if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


    //-- 2. Read the video stream
    capture = cvCaptureFromCAM( -1 );
    if( capture )
      {
        while( true )
        {
          frame = cvQueryFrame( capture );

      //-- 3. Apply the classifier to the frame
          if( !frame.empty() )
          { detectAndDisplay( frame ); }
          else
          { printf(" --(!) No captured frame -- Break!"); break; }

          int c = cv::waitKey(10);
          if( (char)c == 'c' ) { break; }
         }
      }


}

cv::String car_cascade_name = "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/haar/coches.xml";
cv::CascadeClassifier car_cascade;
string window_name = "Car Detection";
cv::RNG rng(12345);

    /** @function detectAndDisplay */
    void detectAndDisplay( cv::Mat frame )
    {
      std::vector<Rect> faces;
      cv::Mat frame_gray;

      cvtColor( frame, frame_gray, CV_BGR2GRAY );
      cv::equalizeHist( frame_gray, frame_gray );

      //-- Detect faces
      car_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

      for( size_t i = 0; i < faces.size(); i++ )
      {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );



      }
      //-- Show what you got
      imshow( window_name, frame );
    }


