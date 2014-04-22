#include "infokeypoint.h"

InfoKeyPoint::InfoKeyPoint(cv::KeyPoint k, cv::Point p)
{
    this->keyPoint=k;
    this->vectorCentro = p;
}

cv::KeyPoint InfoKeyPoint::getKeyPoint(){
    return this->keyPoint;
}

cv::Point InfoKeyPoint::getVectorCentro(){
    return this->vectorCentro;
}
