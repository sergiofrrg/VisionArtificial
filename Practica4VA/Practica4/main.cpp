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
#include <filtro.h>

using namespace std;
using namespace cv;
using namespace libfacerec;

int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

cv::String car_cascade_name = "haar/coches.xml";
cv::CascadeClassifier car_cascade;
const float minimoArea = 105.0 / 24893.0;
const float maximoArea = 300.0 / 24893.0;
const int tamanioResize=10;

Rect detectAndDisplay(cv::Mat frame) {
    std::vector<Rect> cars;

    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    //-- Detect cars
    car_cascade.detectMultiScale(frame_gray, cars, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    Rect coche;

    if (cars.size() == 0) {
        return coche;
    }

    coche = cars[0];

    cout << "Número de coches detectados: " << cars.size() << endl;

    if (cars.size() > 1) {
        for (int i = 1; i < cars.size(); ++i) {
            if (cars[i].area() > coche.area())
                coche = cars[i];
        }
    }
    rectangle(frame, coche, cv::Scalar(255, 255, 255), 2, 8);
    imshow("", frame);
    return coche;
}

////FILTRO 1

//vector<vector<Point> > filtroArea(vector<vector<Point> > contours, cv::Mat frame) {
//    // 1er filtro: descartamos caracteres por su area
//    vector<vector<Point> > newContours;
//    for (int i = 0; i < contours.size(); ++i) {
//        if ((cv::boundingRect(contours.at(i)).area() >= frame.cols * frame.rows * minimoArea)&&(cv::boundingRect(contours.at(i)).area() <= frame.cols * frame.rows * maximoArea))
//            newContours.push_back(contours.at(i));
//        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
//    }
//    return newContours;
//}

////FILTRO 2

//vector<vector<Point> > filtroProporcion(vector<vector<Point> > contours) {
//    vector<vector<Point> > newContours;
//    for (int i = 0; i < contours.size(); ++i) {
//        if ((cv::boundingRect(contours.at(i)).height) > (cv::boundingRect(contours.at(i)).width))
//            newContours.push_back(contours.at(i));
//        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
//    }
//    return newContours;
//}

////FILTRO 3

//vector<vector<Point> > filtroSeparacion(vector<vector<Point> > contornos, cv::Mat frame) {
//    vector<vector<Point> > newContours;
//    for (int i = 0; i < contornos.size(); i++) {
//        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
//        for (int j = 0; j < contornos.size(); j++) {
//            cv::Rect rectAuxCompara = cv::boundingRect(contornos.at(j));
//            if ((std::abs(rectAux.x - rectAuxCompara.x) < (frame.cols / 10)) && (j != i)) {
//                newContours.push_back(contornos.at(i));
//                break;
//            }
//        }
//    }
//    return newContours;
//}

////FILTRO 4

//vector<vector<Point> > filtroPosicion(vector<vector<Point> > contornos, cv::Mat frame) {
//    vector<vector<Point> > newContours;
//    for (int i = 0; i < contornos.size(); i++) {
//        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
//        float centro = frame.cols / 2;
//        if ((rectAux.x >= centro - (frame.cols / 4)) && (rectAux.x <= centro + (frame.cols / 4)))
//            newContours.push_back(contornos.at(i));
//    }
//    return newContours;
//}

////FILTRO 5

//vector<vector<Point> > filtroOrdenacion(vector<vector<Point> > contours) {
//    vector<vector<Point> > newContours;
//    std::vector<int> equises;
//    for (int i = 0; i < contours.size(); ++i) {
//        equises.push_back(cv::boundingRect(contours.at(i)).x);
//    }
//    std::sort(equises.begin(), equises.end());
//    for (int i = 0; i < equises.size(); ++i) {
//        for (int j = 0; j < contours.size(); j++) {
//            if (cv::boundingRect(contours.at(j)).x == equises.at(i))
//                newContours.push_back(contours.at(j));
//        }
//    }
//    return newContours;
//}

//METODO RESIZER: Convierte los dígitos a tamanioResizextamanioResize manteniendo el ratio de aspecto

cv::Mat_<float> resizer(cv::Mat binaryDigit){

    Size size;
    cv::Mat resizedDigit;
    if (((binaryDigit.cols*tamanioResize) / binaryDigit.rows) < 1)
        size = Size (1, tamanioResize);
    else
        size = Size ((binaryDigit.cols * tamanioResize) / binaryDigit.rows, tamanioResize);

    Size size2(tamanioResize, tamanioResize);

    cv::resize(binaryDigit, resizedDigit, size, 0, 0, cv::INTER_LINEAR);

    Point puntoOrigen((tamanioResize - resizedDigit.cols) / 2, 0);

    Rect roi(puntoOrigen, Size(resizedDigit.cols, resizedDigit.rows));


    cv::Mat matriz = cv::Mat::ones(size2, resizedDigit.type())*255;

    resizedDigit.copyTo(matriz(roi));
    resizedDigit = matriz;

    return resizedDigit/255.0;
}

cv::Mat_<float> matrizAFila(cv::Mat_<float> floatMat){
    cv::Mat_<float> fila;

    cv::MatIterator_<float> it;
    for (it = floatMat.begin(); it != floatMat.end(); ++it) {

        fila.push_back(*it);
    }
    return fila.t();
}

//cv::Mat borrarBordes (cv::Mat digito){
//    int arriba=0;
//    int abajo=0;
//    int izq=0;
//    int der=0;
//    int col=0;
//    int row=0;
//    bool negro=false;
//    //cout << "Primer While" << endl;
//    while(!negro&&row<digito.rows){
//        uchar* p=digito.ptr(row);
//        while(!negro&&col<digito.cols){
//            if((int)*p==0){
//                negro=true;
//            }
//            p++;
//            ++col;
//        }
//        ++row;
//        col=0;

//    }
//    if(negro){
//        arriba=row-1;
//    }

//    row=digito.rows-1;
//    col=0;
//    negro=false;
//    while(!negro&&row>=0){
//        uchar* p=digito.ptr(row);
//        while(!negro&&col<digito.cols){
//            if((int)*p==0){
//                negro=true;
//            }
//            p++;
//            ++col;
//        }
//        --row;
//        col=0;
//    }
//    if(negro){
//        abajo=row+1;
//    }

//    row = 0;
//    col = 0;
//    negro=false;
//    uchar* p=digito.ptr(row);
//    while(!negro&&col<digito.cols){
//        while(!negro&&row<digito.rows){
//            p=digito.ptr(row);
//            p+=col;
//            if((int)*p==0){
//                negro=true;
//            }
//            ++row;
//        }
//        //p++;
//        ++col;
//        row = 0;
//    }
//    if(negro){
//        izq=col-1;
//    }

//    row = 0;
//    col = digito.cols-1;
//    negro=false;
//    p=digito.ptr(row);
//    while(!negro&&col>=0){
//        while(!negro&&row<digito.rows){
//            p=digito.ptr(row);
//            p+=col;
//            if((int)*p==0)
//            {
//                negro=true;
//            }
//            ++row;
//        }
//        //p--;
//        --col;
//        row = 0;
//    }
//    if(negro){
//        der=col+1;
//    }

//    digito = digito.rowRange(arriba, abajo);
//    digito = digito.colRange(izq, der);
//    return digito;

//}

int main(int argc, char **argv) {
    if (argc != 1){
        cout << "ERROR: Debe introducir la ruta de la imagen o vídeo a analizar" << endl;
        exit (0);
    }

    string matricula;

    string ruta1;
    cv::Rect rectangulo;
    //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
    //stringstream ss;
    //Dirección de labSergio
    //ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << "1" << ".jpg";
    //ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/TestCars/Test/test" << "25" << ".jpg";
    //Dirección de labAza
    //ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << 17 << ".jpg";
    ruta1 = argv[1];

    //////////////////////////////////////
    // APRENDIZAJE //////////////////////
    /////////////////////////////////////

    //PARTE 2 //////////////////////////////////////////////////////////////////////////////////
    string ruta;
    cv::Mat_<uchar> digito;
    string clases [37] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "ESP", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    vector<string> e;
    vector<int> eNteros;
    cv::Mat_<float> mCaracteristicas;

    int nClase = 0;

    for (int i = 1; i <= 9250; i++) {

        stringstream ss;

        int aux = i % 250;
        if (aux == 0)
            aux = 250;

        ss << "Digitos/" << clases[nClase] << "_" << aux << ".jpg";

        ruta = ss.str();
        digito = cv::imread(ruta, 0);

        //Binary image
        cv::Mat binaryDigit(digito.size(), digito.type());

        //Apply thresholding
        cv::threshold(digito, binaryDigit, 200, 255, cv::THRESH_BINARY);

        binaryDigit=Filtro::borrarBordes(binaryDigit.clone());

        //Se reescala la imagen del dígito a tamanioResizextamanioResize
        cv::Mat_<float> floatMat = resizer(binaryDigit);

        e.push_back(clases[nClase]);
        eNteros.push_back(nClase);
        mCaracteristicas.push_back((cv::Mat_<float>)matrizAFila(floatMat));

        if (i % 250 == 0)
            nClase++;

    }

    //Creamos el LDA

    LDA lda(mCaracteristicas, eNteros);
    cv::Mat_<float> cr;
    cr = lda.project(mCaracteristicas);

    //Creamos el índice de flann

    cv::flann::Index i(cr, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);


    ///////////////////////////////////////////////////////
    // TESTING
    ///////////////////////////////////////////////////////
    string direccion = ruta1;
    cv::VideoCapture vCap(direccion);
    cv::Mat frame;
    //cv::Mat frame=cv::imread(ruta1, CV_LOAD_IMAGE_COLOR);

    //-- 1. Load the cascades
    if (!car_cascade.load(car_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    //-- 2. Read the video stream

    if (vCap.isOpened()) {
        while (true) {
            vCap.read(frame);

            if (!frame.empty()) {
                rectangulo = detectAndDisplay(frame);
                if (!rectangulo.area() == 0) {

                    frame = frame.rowRange(rectangulo.y + rectangulo.height / 2, rectangulo.y + rectangulo.height);
                    frame = frame.colRange(rectangulo.x, rectangulo.x + rectangulo.width);

                    cout << "Nuevo Frame" << endl;

                    cv::Mat imageMat = frame;

                    //Grayscale matrix
                    cv::Mat grayscaleMat(imageMat.size(), CV_8U);

                    //Convert BGR to Gray
                    cv::cvtColor(imageMat, grayscaleMat, CV_BGR2GRAY);

                    //Binary image
                    cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

                    //Apply thresholding
                    cv::adaptiveThreshold(grayscaleMat, binaryMat, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 10);

                    cv::Mat imagenUmbralizadaCopia = binaryMat.clone();

                    vector<vector<Point> > contours;
                    vector<cv::Vec4i> hierarchy;

                    findContours(binaryMat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
                    vector<vector<Point> > newContours;

                    //PASAMOS LOS FILTROS
                    newContours = Filtro::filtroArea(contours, frame);
                    newContours = Filtro::filtroProporcion(newContours);
                    newContours = Filtro::filtroSeparacion(newContours, frame);
                    newContours = Filtro::filtroPosicion(newContours, frame);
                    newContours = Filtro::filtroOrdenacion(newContours);

                    cout << "Número de contornos final: " << newContours.size() << endl;

                    for (int j = 0; j < newContours.size(); ++j) {

                        cv::Mat imageAux = imagenUmbralizadaCopia;
                        Rect rectAux = cv::boundingRect(newContours.at(j));
                        imageAux = imageAux.rowRange(rectAux.y, rectAux.y + rectAux.height);
                        imageAux = imageAux.colRange(rectAux.x, rectAux.x + rectAux.width);

                        cv::Mat_<float> floatMat = resizer(imageAux);

                        cv::Mat_<float> matReduced = lda.project(matrizAFila(floatMat));

                        int vectorVotacion[37];
                        for (int k = 0; k < 37; ++k) {
                            vectorVotacion[k] = 0;
                        }

                        int k = 40;
                        cv::Mat_<int> indices;
                        cv::Mat dist;
                        i.knnSearch(matReduced, indices, dist, k);

                        for (cv::Mat_<int>::iterator it = indices.begin(); it != indices.end(); it++) {

                            vectorVotacion[eNteros[(*it)]]++;
                        }

                        int valorMayor = 0;
                        int voto;
                        for (int cont = 0; cont < 37; ++cont) {
                            if (vectorVotacion[cont] > valorMayor) {
                                valorMayor = vectorVotacion[cont];
                                voto = cont;
                            }

                        }

                        //cout << clases[voto] << endl;
                        matricula += clases[voto];

                    }

                    string aux = matricula;

                    cout << "Matricula encontrada: " << aux << endl << "--------------------------------------" << endl;
                    matricula = "";
                    waitKey(0);

                } else {
                    cout << "--(!) No se han detectado coches" << endl << "--------------------------------------" << endl;
                }
            } else {
                cout << " --(!) No se encuentran más imágenes o frames -- Break!" << endl;
                break;
            }

            int c = cv::waitKey(10);
            if ((char) c == 'c') {
                break;
            }
        }
    } else {
        cout << "No se está leyendo el vídeo" << endl;
        return -1;
    }
}
