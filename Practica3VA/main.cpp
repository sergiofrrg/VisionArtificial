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

cv::String car_cascade_name = "haar/coches.xml";
cv::CascadeClassifier car_cascade;
string window_name = "Detección en vídeo";
cv::RNG rng(12345);

int main(int argc, char **argv) {
    cv::Mat_<uchar> image;
    cv::ORB orb(10, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
    cv::Mat_<uchar> descriptoresImagen;
    cv::Mat_< cv::Mat_<uchar> > conjuntoDescriptores;
    vector<cv::KeyPoint> kp;
    string ruta;
    cv::Point centro;

    cv::Point tamanioImagenesAprendizaje;

    vector<InfoKeyPoint> listaInfoKeyPoints;

    for (int i = 1; i <= 48; i++) {
        if (argc < 2){
            cout << "ERROR: Debe poner la/s ruta/s de la 1.imagen (y 2.vídeo)" << endl;
            exit(0);
        }

        stringstream ss;
        ss << "LearningCars/training_frontal/frontal_" << i << ".jpg";
        ruta = ss.str();
        image = cv::imread(ruta, 0);

        //HALLAMOS Y GUARDAMOS LOS KEYPOINTS DE LA IMAGEN EN EL VECTOR kp Y LOS DESCRIPTORES EN LA MATRIZ descriptoresImagen
        orb.detect(image, kp);
        orb.compute(image, kp, descriptoresImagen);

        //GUARDAMOS EL CENTRO DE LA IMAGEN
        centro = cv::Point(image.cols / 2, image.rows / 2);

        //ALMACENAMOS LA DIRECCIÓN AL CENTRO DE CADA KEYPOINT DE LA IMAGEN Y EL PROPIO KEYPOINT
        //en vectorInfoKP (y este a su vez en conjuntoInfoKeyPoints)
        for (int j = 0; j < kp.size(); j++) {
            listaInfoKeyPoints.push_back(InfoKeyPoint(kp.at(j),
                    cv::Point(centro.x - kp.at(j).pt.x,
                    centro.y - kp.at(j).pt.y)));
        }

        //ALMACENAMOS LA MATRIZ DE DESCRIPTORES DE LA IMAGEN i EN LA MATRIZ conjuntoDescriptores
        conjuntoDescriptores.push_back(descriptoresImagen);
    }

    tamanioImagenesAprendizaje.x = image.cols;
    tamanioImagenesAprendizaje.y = image.rows;

    //CREAMOS EL ÍNDICE i PARA conjuntoDescriptores
    cv::flann::Index i(conjuntoDescriptores, cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_HAMMING);

    //CARGAMOS IMAGEN TEST
    cv::Mat_<uchar> image3;

    image3 = cv::imread(argv[1], 0);

    //HALLAMOS LOS KEYPOINTS Y DESCRIPTORES DE LA IMAGEN DE TEST
    orb.detect(image3, kp);
    orb.compute(image3, kp, descriptoresImagen);

    //BUSCAMOS LOS k VECINOS MÁS CERCANOS A LOS DESCRIPTORES DE LA IMAGEN DE TEST EN descriptoresImagen
    int k = 10;
    cv::Mat_<int> indices;
    cv::Mat dist;
    i.knnSearch(descriptoresImagen, indices, dist, k);

    //CREAMOS MATRIZ DE VOTACIÓN DEL TAMAÑO DE LA IMAGEN/bajaRes Y LO LLENAMOS DE 0s
    int bajaRes = 10;
    int matVotacion[image3.cols / bajaRes][image3.rows / bajaRes];
    for (int i = 0; i < image3.cols / bajaRes; i++) {
        for (int j = 0; j < image3.rows / bajaRes; j++) {
            matVotacion[i][j] = 0;
        }
    }

    double escalaImagenTest = kp.at(0).size;

    //OBTENEMOS LOS INFOKEYPOINTS DE APRENDIZAJE CORRESPONDIENTES A LOS DESCRIPTORES DE
    //APRENDIZAJE MÁS PARECIDOS A LOS DE LA IMAGEN TEST

    int contadorKP = 0;
    int contadorColumna = 1;

    for (cv::Mat_<int>::iterator it = indices.begin(); it != indices.end(); it++) {
        if (contadorColumna > k) {
            contadorKP++;
            contadorColumna = 1;
        }

        //Guardamos el keypoint de la imagen aprendizaje k y su dircentro
        InfoKeyPoint infoKpAux = listaInfoKeyPoints.at(*it);
        cv::KeyPoint kpAux = infoKpAux.getKeyPoint();
        cv::Point dirCentroAux = infoKpAux.getVectorCentro();

        //Modificamos la escala de donde supuestamente está el centro comparando la de la
        //imagen test con la de la imagen de aprendizaje (de cualquiera de sus keypoints)
        double escalaAprendizajeImagenK = kpAux.size;
        double reescalador = escalaImagenTest / escalaAprendizajeImagenK;
        dirCentroAux.x = dirCentroAux.x * (reescalador);
        dirCentroAux.y = dirCentroAux.y * (reescalador);

        listaInfoKeyPoints.at(*it).setDifEscala(reescalador);

        //Le sumamos el dirCentroAux al keyPoint actual de la imagen test para obtener el
        //supuesto centro
        cv::Point votoCentro;
        votoCentro.x = kp.at(contadorKP).pt.x + dirCentroAux.x;
        votoCentro.y = kp.at(contadorKP).pt.y + dirCentroAux.y;

        listaInfoKeyPoints.at(*it).setVoto(votoCentro);

        //Votamos dividiendo las coordenadas entre bajaRes si no busca en un punto fuera de la imagen
        if ((votoCentro.x <= image3.cols) && (votoCentro.y <= image3.rows))
            matVotacion[votoCentro.x / bajaRes][votoCentro.y / bajaRes]++;

        contadorColumna++;
    }


    //DESPUÉS DE LA VOTACIÓN, BUSCAMOS EL PUNTO QUE TIENE MÁS VOTOS

    int valorMayor = 0;
    cv::Point centroFinal;
    for (int j = 0; j < image3.rows / bajaRes; j++) {
        for (int i = 0; i < image3.cols / bajaRes; i++) {
            if (matVotacion[i][j] > valorMayor) {
                valorMayor = matVotacion[i][j];
                centroFinal.x = i*bajaRes;
                centroFinal.y = j*bajaRes;
            }
        }
    }

    cv::circle(image3,
            centroFinal,
            3.0,
            cv::Scalar(255, 0, 255),
            2,
            8);

    cv::imshow("imagen", image3);
    cv::waitKey();

    if (argc==2)
        exit(0);

    //AQUI COMIENZA LO DEL HAAR

    void detectAndDisplay(cv::Mat frame);

    string direccion;
    if (argc == 3)
        direccion = argv[2];
    else
        direccion = argv[1];
    cv::VideoCapture vCap(direccion);
    cv::Mat frame;

    //-- 1. Load the cascades
    if (!car_cascade.load(car_cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    };

    //-- 2. Read the video stream
    if (vCap.isOpened()) {
        while (true) {
            vCap.read(frame);

            //-- 3. Apply the classifier to the frame
            if (!frame.empty()) {
                detectAndDisplay(frame);
            } else {
                printf(" --(!) No captured frame -- Break!");
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

void detectAndDisplay(cv::Mat frame) {
    std::vector<Rect> cars;
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    car_cascade.detectMultiScale(frame_gray, cars, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    cout << "Número de coches detectados: " << cars.size() << endl;

    for (size_t i = 0; i < cars.size(); i++) {
        rectangle(frame, cars[i], cv::Scalar(255, 255, 255), 2, 8);
    }

    imshow(window_name, frame);

}
