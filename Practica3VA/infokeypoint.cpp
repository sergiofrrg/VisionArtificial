#include "infokeypoint.h"

InfoKeyPoint::InfoKeyPoint(cv::KeyPoint k, cv::Point p)
{
    this->keyPoint=k;
    this->vectorCentro = p;
    this->difEscala = 1;
}

cv::KeyPoint InfoKeyPoint::getKeyPoint(){
    return this->keyPoint;
}

cv::Point InfoKeyPoint::getVectorCentro(){
    return this->vectorCentro;
}

void InfoKeyPoint::setVoto(cv::Point p){
    this->voto = p;
}

void InfoKeyPoint::setDifEscala(double e){
    this->difEscala = e;
}

cv::Point InfoKeyPoint::getVoto(){
    return this->voto;
}

double InfoKeyPoint::getDifEscala(){
    return this->difEscala;
}
