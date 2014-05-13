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

cv::String car_cascade_name = "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/haar/coches.xml";
cv::CascadeClassifier car_cascade;
string window_name = "Car Detection";
const float minimoArea=105.0/24893.0;
const float maximoArea=300.0/24893.0;
//const int tamanioResize=20;


/** @function detectAndDisplay */
Rect detectAndDisplay( cv::Mat frame)
{
  std::vector<Rect> cars;

  //imshow("a", frame);
  //cv::waitKey();
  cv::Mat frame_gray;

  cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
  cv::equalizeHist( frame_gray, frame_gray );

  //-- Detect cars
  car_cascade.detectMultiScale( frame_gray, cars, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
  Rect coche=cars[0];

  cout << "Número de coches detectados: " << cars.size() << endl;

  if (cars.size()>1){
      for (int i=1; i<cars.size(); ++i){
          if (cars[i].area()>coche.area())
              coche=cars[i];
      }
  }

  int i=0;
    Point center( coche.x + coche.width*0.5, coche.y + coche.height*0.5 );
    //ellipse( frame, center, Size( cars[i].width*0.5, cars[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    //rectangle(frame, cars[i], cv::Scalar (255, 255, 255), 2, 8 );
    cout << center << endl;
    imshow("",frame);
    return coche;
}

//FILTRO 1
vector<vector<Point> > filtroArea(vector<vector<Point> > contours, cv::Mat frame){
    // 1er filtro: descartamos caracteres por su area
    vector<vector<Point> > newContours;
    for(int i=0; i<contours.size(); ++i){
        if((cv::boundingRect(contours.at(i)).area()>=frame.cols*frame.rows*minimoArea)&&(cv::boundingRect(contours.at(i)).area()<=frame.cols*frame.rows*maximoArea))
            newContours.push_back(contours.at(i));
        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
    }
    return newContours;
}

//FILTRO 2
vector<vector<Point> > filtroProporcion(vector<vector<Point> > contours){
    vector<vector<Point> > newContours;
    for(int i=0; i<contours.size(); ++i){
        if((cv::boundingRect(contours.at(i)).height) > (cv::boundingRect(contours.at(i)).width))
            newContours.push_back(contours.at(i));
        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
    }
    return newContours;
}

//FILTRO 3
vector<vector<Point> > filtroSeparacion(vector<vector<Point> > contornos, cv::Mat frame){
    vector<vector<Point> > newContours;
    for (int i=0; i<contornos.size(); i++){
        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
        for (int j = 0; j<contornos.size(); j++){
            cv::Rect rectAuxCompara = cv::boundingRect(contornos.at(j));
            if ((std::abs(rectAux.x - rectAuxCompara.x) < (frame.cols/10) ) && (j!=i)){
                newContours.push_back(contornos.at(i));
                break;
            }
        }
    }
    return newContours;
}

//FILTRO 4
vector<vector<Point> > filtroPosicion(vector<vector<Point> > contornos, cv::Mat frame){
    vector<vector<Point> > newContours;
    for (int i=0; i<contornos.size(); i++){
        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
        float centro = frame.cols/2;
        if ((rectAux.x >= centro-(frame.cols/4)) && (rectAux.x <= centro+(frame.cols/4)))
            newContours.push_back(contornos.at(i));
    }
    return newContours;
}

//FILTRO 5
vector<vector<Point> > filtroOrdenacion(vector<vector<Point> > contours){
    vector<vector<Point> > newContours;
    std::vector<int> equises;
    for(int i=0; i<contours.size(); ++i){
        equises.push_back(cv::boundingRect(contours.at(i)).x);
    }
    std::sort(equises.begin(), equises.end());
    for(int i=0; i<equises.size(); ++i){
        for (int j=0; j<contours.size(); j++){
            if (cv::boundingRect(contours.at(j)).x == equises.at(i))
                newContours.push_back(contours.at(j));
        }
    }
    return newContours;
}

int main()
{
   string matricula;
   string ruta;
   string ruta1;
   cv::Rect rectangulo;
   //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
   stringstream ss;
   //Dirección de labSergio
   //ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << "1" << ".jpg";
   ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/TestCars/Test/test" << "28" << ".jpg";
   //Dirección de labAza
   //ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << 17 << ".jpg";
   ruta1 = ss.str();



   //AQUI COMIENZA LO DEL HAAR
   //CvCapture* capture;
   string direccion= "/home/sferrer/Documentos/Videos/video2.wmv";
   cv::VideoCapture vCap (direccion);
   cv::Mat frame;
   //cv::Mat frame=cv::imread(ruta1, CV_LOAD_IMAGE_COLOR);

   //-- 1. Load the cascades
   if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream
   //capture = cvCaptureFromAVI("/home/sferrer/Documentos/Videos/video1.wmv");
    if( vCap.isOpened() )
      {
        while( true )
        {
          vCap.read(frame);


         if( !frame.empty() )
         {
            rectangulo=detectAndDisplay( frame );
            cout << "Rect: "<< rectangulo.area()/2 << endl;


            cout << rectangulo.x << " " << rectangulo.y << " " << rectangulo.x+rectangulo.width << " " << rectangulo.y+rectangulo.height << endl;
            cout << frame.cols << " " << frame.rows << endl;

                //Our color image
         //   cv::Rect  rect(rectangulo.x, rectangulo.y,rectangulo.x+rectangulo.width, rectangulo.y+rectangulo.height);
         //   cv::Mat imageMat  (frame(rect));

            frame=frame.rowRange(rectangulo.y+rectangulo.height/2,rectangulo.y+rectangulo.height);
            frame=frame.colRange(rectangulo.x, rectangulo.x+rectangulo.width);

            cout << "Frame: " << frame.cols*frame.rows << endl;


            //imshow( window_name, frame );
            //waitKey(0);

             cv::Mat imageMat=frame;



                //imshow( "Contours", imageMat );

                //cv::waitKey(0);


                //Grayscale matrix
                cv::Mat grayscaleMat (imageMat.size(), CV_8U);

                //Convert BGR to Gray
                cv::cvtColor( imageMat, grayscaleMat, CV_BGR2GRAY );

                //Binary image
                cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

                //Apply thresholding
                //cv::threshold(grayscaleMat, binaryMat, 128, 255, cv::THRESH_BINARY);
                cv::adaptiveThreshold(grayscaleMat, binaryMat, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 10);
                //Show the results
                //cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);

                //cv::imshow("Binary Image", binaryMat);               //Imagen del coche binarizada

                //cv::waitKey(0);
                 cv::Mat imagenUmbralizadaCopia=binaryMat.clone();

                 vector<vector<Point> > contours;
                 vector<cv::Vec4i> hierarchy;



                 /// Find contours
                 findContours( binaryMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
                 vector<vector<Point> > newContours;

                 //cout << (rectangulo.area()/2)*minimoArea << endl;

                 //PASAMOS LOS FILTROS
                 newContours = filtroArea(contours, frame);
                 newContours = filtroProporcion(newContours);
                 newContours = filtroSeparacion(newContours, frame);
                 newContours = filtroPosicion(newContours, frame);
                 newContours = filtroOrdenacion(newContours);

                 cout << newContours.size() << endl;


                 /// Draw contours
                 Mat drawing = Mat::zeros( binaryMat.size(), CV_8UC3 );
                 for( int i = 0; i< newContours.size(); i++ )
                    {
                      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                      drawContours( drawing, newContours, i, color, 2, 8, hierarchy, 0, Point() );
                    }


                 /// Show in a window

                 //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );      //Se dibujan los contornos encontrados
                 //imshow( "Contours", drawing );

                 //cv::waitKey(0);

                 //PARTE 2 //////////////////////////////////////////////////////////////////////////////////
                 cv::Mat_<uchar> digito;
                 string clases [37] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ESP", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
                 vector<string> e;
                 vector<int> eNteros;
                 cv::Mat_<float> mCaracteristicas;
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

                     //Binary image
                     cv::Mat binaryDigit(digito.size(), digito.type());

                     //Apply thresholding

                     cv::adaptiveThreshold(digito, binaryDigit, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 10);

                     //imshow ("", binaryDigit);
                     //waitKey(0);

                     //cv::threshold(digito, binaryDigit, 100, 255, cv::THRESH_BINARY);

                     //Se reescala la imagen del dígito a 10x10

                     //cv::Mat_<int> resizedDigit(10, 10, DataType<int>::type);

                     cv::Mat resizedDigit;

                     Size size((binaryDigit.cols*10)/binaryDigit.rows,10);

                     Size size2 (10,10);

                     cv::resize(binaryDigit,resizedDigit,size,0,0,cv::INTER_LINEAR );

                     Point puntoOrigen ((10-resizedDigit.cols)/2,0);

                     Rect roi(puntoOrigen,Size (resizedDigit.cols, resizedDigit.rows));


                     cv::Mat matriz=cv::Mat::ones(size2, resizedDigit.type())*255;

                     resizedDigit.copyTo(matriz(roi));


                     //imshow(" ", matriz);
                     //waitKey(0);

                     resizedDigit=matriz;


                     cv::Mat_<float> floatMat=resizedDigit/255.0;


                     //cout << floatMat.row(1) << endl;

                     //Centrar los Digitos ¿?

                     //Convertir las imágenes a matrices de 1x100


                     cv::Mat_<float> fila;

                     cv::MatIterator_<float> it;
                     for( it = floatMat.begin(); it != floatMat.end(); ++it){

                         fila.push_back(*it);


                     }



                     e.push_back(clases[nClase]);
                     eNteros.push_back(nClase);
                     mCaracteristicas.push_back((cv::Mat_<float>)fila.t());

                     if (i % 250 == 0)
                         nClase++;

                 }

                 LDA lda(mCaracteristicas, eNteros);
                 cv::Mat_<float> cr;
                 cr = lda.project(mCaracteristicas);

                 cout << cr.rows << " " << cr.cols << endl;
                 cout << cr.row(0);

                 //Creamos el índice de flann

                 cv::flann::Index i (cr, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);



                 for (int j=0; j < newContours.size(); ++j){

                     cv::Mat imageAux=imagenUmbralizadaCopia;
                     Rect rectAux=cv::boundingRect(newContours.at(j));
                     imageAux=imageAux.rowRange(rectAux.y,rectAux.y+rectAux.height);
                     imageAux=imageAux.colRange(rectAux.x, rectAux.x+rectAux.width);

                     cv::Mat resizedDigit;

                     Size size((imageAux.cols*10)/imageAux.rows,10);

                     Size size2 (10,10);

                     cv::resize(imageAux,resizedDigit,size,0,0,cv::INTER_LINEAR );

                     Point puntoOrigen ((10-resizedDigit.cols)/2,0);

                     Rect roi(puntoOrigen,Size (resizedDigit.cols, resizedDigit.rows));


                     cv::Mat matriz=cv::Mat::ones(size2, resizedDigit.type())*255;

                     resizedDigit.copyTo(matriz(roi));


                     //imshow(" ", matriz);
                     //waitKey(0);

                     resizedDigit=matriz;


                     cv::Mat_<float> floatMat=resizedDigit/255.0;

                     //imshow ("", floatMat);
                     //waitKey(0);


                     cv::Mat_<float> fila;

                     cv::MatIterator_<float> it;
                     for( it = floatMat.begin(); it != floatMat.end(); ++it){

                         fila.push_back(*it);


                     }

                     cv::Mat_<float> matReduced = lda.project(fila.t());
                     //cout << matReduced.rows << " " << matReduced.cols << endl;

                     int vectorVotacion[37];
                     for(int k=0; k<37; ++k){
                         vectorVotacion[k]=0;
                     }

                     int k = 40;
                     cv::Mat_<int> indices;
                     cv::Mat dist;
                     i.knnSearch(matReduced, indices, dist, k);

                     for(cv::Mat_<int>::iterator it = indices.begin(); it!=indices.end(); it++){

                         vectorVotacion[eNteros[(*it)]]++;
                     }

                     int valorMayor=0;
                     int voto;
                     for(int cont=0; cont<37; ++cont){
                         if(vectorVotacion[cont]>valorMayor){
                             valorMayor=vectorVotacion[cont];
                             voto=cont;
                         }

                     }

                     //cout << clases[voto] << endl;
                     matricula+=clases[voto];

                 }

                 cv::Point pt1;

                 //cv::InitFont( &font, CV_FONT_VECTOR0, 0.5, 0.5, 0, 2.0, CV_AA);	//Inicializamos el código fuente

                 pt1.x = 100;

                 pt1.y = 100;

                 cv::Mat foticuen=cv::imread(ruta1, CV_LOAD_IMAGE_COLOR);

                 cv::putText(foticuen, matricula, pt1, 2, 3, 255);

                 //cv::imshow("Ejemplo5", foticuen );
                 //cv::waitKey(0);

                 string aux=matricula;

                 cout << aux << endl;

                 aux="";
         }
         else
         { printf(" --(!) No captured frame -- Break!"); break; }

         int c = cv::waitKey(10);
         if( (char)c == 'c' ) { break; }
        }
   }else {
       cout << "No se está leyendo el vídeo" << endl;
       return -1;
   }






}
