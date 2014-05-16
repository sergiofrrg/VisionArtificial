#include "filtro.h"
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

Filtro::Filtro()
{
}

vector<vector<Point> >Filtro::filtroArea(vector<vector<Point> > contours, cv::Mat frame, float minimoArea, float maximoArea) {
    // 1er filtro: descartamos caracteres por su area
    vector<vector<Point> > newContours;
    for (int i = 0; i < contours.size(); ++i) {
        if ((cv::boundingRect(contours.at(i)).area() >= frame.cols * frame.rows * minimoArea)&&(cv::boundingRect(contours.at(i)).area() <= frame.cols * frame.rows * maximoArea))
            newContours.push_back(contours.at(i));
        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
    }
    return newContours;
}


vector<vector<Point> >Filtro::filtroProporcion(vector<vector<Point> > contours) {
    vector<vector<Point> > newContours;
    for (int i = 0; i < contours.size(); ++i) {
        if ((cv::boundingRect(contours.at(i)).height) > (cv::boundingRect(contours.at(i)).width))
            newContours.push_back(contours.at(i));
        //cout << cv::boundingRect(contours.at(i)).area()<< endl;
    }
    return newContours;
}

vector<vector<Point> >Filtro::filtroSeparacion(vector<vector<Point> > contornos, cv::Mat frame) {
    vector<vector<Point> > newContours;
    for (int i = 0; i < contornos.size(); i++) {
        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
        for (int j = 0; j < contornos.size(); j++) {
            cv::Rect rectAuxCompara = cv::boundingRect(contornos.at(j));
            if ((std::abs(rectAux.x - rectAuxCompara.x) < (frame.cols / 10)) && (j != i)) {
                newContours.push_back(contornos.at(i));
                break;
            }
        }
    }
    return newContours;
}

vector<vector<Point> >Filtro::filtroPosicion(vector<vector<Point> > contornos, cv::Mat frame) {
    vector<vector<Point> > newContours;
    for (int i = 0; i < contornos.size(); i++) {
        cv::Rect rectAux = cv::boundingRect(contornos.at(i));
        float centro = frame.cols / 2;
        if ((rectAux.x >= centro - (frame.cols / 4)) && (rectAux.x <= centro + (frame.cols / 4)))
            newContours.push_back(contornos.at(i));
    }
    return newContours;
}

vector<vector<Point> >Filtro::filtroOrdenacion(vector<vector<Point> > contours) {
    vector<vector<Point> > newContours;
    std::vector<int> equises;
    for (int i = 0; i < contours.size(); ++i) {
        equises.push_back(cv::boundingRect(contours.at(i)).x);
    }
    std::sort(equises.begin(), equises.end());
    for (int i = 0; i < equises.size(); ++i) {
        for (int j = 0; j < contours.size(); j++) {
            if (cv::boundingRect(contours.at(j)).x == equises.at(i))
                newContours.push_back(contours.at(j));
        }
    }
    return newContours;
}

cv::Mat Filtro::borrarBordes (cv::Mat digito){
    int arriba=0;
    int abajo=0;
    int izq=0;
    int der=0;
    int col=0;
    int row=0;
    bool negro=false;
    //cout << "Primer While" << endl;
    while(!negro&&row<digito.rows){
        uchar* p=digito.ptr(row);
        while(!negro&&col<digito.cols){
            if((int)*p==0){
                negro=true;
            }
            p++;
            ++col;
        }
        ++row;
        col=0;

    }
    if(negro){
        arriba=row-1;
    }

    row=digito.rows-1;
    col=0;
    negro=false;
    while(!negro&&row>=0){
        uchar* p=digito.ptr(row);
        while(!negro&&col<digito.cols){
            if((int)*p==0){
                negro=true;
            }
            p++;
            ++col;
        }
        --row;
        col=0;
    }
    if(negro){
        abajo=row+1;
    }

    row = 0;
    col = 0;
    negro=false;
    uchar* p=digito.ptr(row);
    while(!negro&&col<digito.cols){
        while(!negro&&row<digito.rows){
            p=digito.ptr(row);
            p+=col;
            if((int)*p==0){
                negro=true;
            }
            ++row;
        }
        //p++;
        ++col;
        row = 0;
    }
    if(negro){
        izq=col-1;
    }

    row = 0;
    col = digito.cols-1;
    negro=false;
    p=digito.ptr(row);
    while(!negro&&col>=0){
        while(!negro&&row<digito.rows){
            p=digito.ptr(row);
            p+=col;
            if((int)*p==0)
            {
                negro=true;
            }
            ++row;
        }
        //p--;
        --col;
        row = 0;
    }
    if(negro){
        der=col+1;
    }

    digito = digito.rowRange(arriba, abajo);
    digito = digito.colRange(izq, der);
    return digito;

}
