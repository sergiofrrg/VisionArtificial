#ifndef FILTRO_H
#define FILTRO_H

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
using namespace cv;

class Filtro
{
public:
    Filtro();
    vector<vector<Point> > filtroArea(vector<vector<Point> > contours, cv::Mat frame, float minimoArea, float maximoArea);
    vector<vector<Point> > filtroProporcion(vector<vector<Point> > contours);
    vector<vector<Point> > filtroSeparacion(vector<vector<Point> > contornos, cv::Mat frame);
    vector<vector<Point> > filtroPosicion(vector<vector<Point> > contornos, cv::Mat frame);
    vector<vector<Point> > filtroOrdenacion(vector<vector<Point> > contours);
    cv::Mat borrarBordes (cv::Mat digito);
};

#endif // FILTRO_H
