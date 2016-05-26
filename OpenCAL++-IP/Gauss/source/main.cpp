#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include <opencv2/opencv.hpp>
#include <functional>
typedef unsigned int COORD_TYPE;

typedef std::array<unsigned char, 2> vec2b;
typedef std::array<unsigned char, 3> vec3b;
typedef std::array<unsigned char, 4> vec4b;


template<class T>
T *loadImage(int size, const std::string& path){
printf("sto qui\n");
    cv::Mat mat= cv::imread(path);

    //int size = mat.rows * mat.cols;
    T* vec = new T [size];
    int linearIndex = 0;

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
            T& bgra = mat.at<T>(i, j);
            vec[linearIndex] = bgra;
        }
    }


    return vec;
}


template<class T>
void saveImage(const T* buffer, const std::string& path){
printf("sto qui save\n");
    
}



int main ()
{
    std::array<COORD_TYPE, 2> coords = { 8, 16 };
    opencal::CALMooreNeighborhood<2> neighbor;

    opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE> calmodel(
      coords,
      &neighbor,
      opencal::calCommon::CAL_SPACE_TOROIDAL,
      opencal::calCommon::CAL_NO_OPT);

    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,
    COORD_TYPE >> calrun(&calmodel, 1, 4, opencal::calCommon::CAL_UPDATE_IMPLICIT);

    opencal::CALSubstate<vec3b, 2, COORD_TYPE> *bgr = calmodel.addSubstate<vec3b>();
    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec3b>)>(loadImage<vec3b>)), "");
    bgr->saveSubstate(*(new std::function<decltype(saveImage<vec3b>)>(saveImage<vec3b>)), "");



}
