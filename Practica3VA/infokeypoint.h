#ifndef INFOKEYPOINT_H
#define INFOKEYPOINT_H

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

class InfoKeyPoint
{
    cv::KeyPoint keyPoint;
    cv::Point vectorCentro;

public:
    InfoKeyPoint(cv::KeyPoint k, cv::Point p);
    cv::KeyPoint getKeyPoint();
    cv::Point getVectorCentro();
};

#endif // INFOKEYPOINT_H
