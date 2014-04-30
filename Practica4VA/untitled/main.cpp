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

using namespace std;

int main()
{
   cv::Mat_<uchar> image;
   string ruta;

       //ss << "/home/sergiofrrg/Documentos/OPENCV/training/frontal_" << i << ".jpg";
       stringstream ss;
       //Dirección de labSergio
       ss << "/home/sferrer/Documentos/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << "1" << ".jpg";

       //Dirección de labAza
       //ss << "/home/aza/Documentos/Universidad/VisionArtificial/EnunciadoP3/LearningCars/training_frontal/frontal_" << i << ".jpg";

       ruta = ss.str();
       image=cv::imread(ruta,0);

}
