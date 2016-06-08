
#ifndef STANDARDCONTRASTSTRETCHINGFILTER_H
#define STANDARDCONTRASTSTRETCHINGFILTER_H


#include <OpenCAL++/calElementaryProcessFunctor.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calSubstate.h>
#include<opencv2/core/core.hpp>


template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE, class PIXELTYPE>
class MinMaxRGBGlobalFunction : public opencal::CALGlobalFunction<DIMENSION,NEIGHBORHOOD,COORDINATE_TYPE>{

    opencal::CALSubstate<PIXELTYPE,DIMENSION>* img;
    double ** minMaxRGB;
    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
public:
    MinMaxRGBGlobalFunction(auto* sbs, double** _minMaxRGB): opencal::CALGlobalFunction<DIMENSION,NEIGHBORHOOD,COORDINATE_TYPE> (opencal::calCommon::CAL_UPDATE_IMPLICIT), img(sbs){

        this->minMaxRGB = _minMaxRGB;

        minMaxRGB[0][0] = img->getElement(0)[0];
        minMaxRGB[0][1] = img->getElement(0)[0];
        minMaxRGB[1][0] = img->getElement(0)[1];
        minMaxRGB[1][1] = img->getElement(0)[1];
        minMaxRGB[2][0] = img->getElement(0)[2];
        minMaxRGB[2][1] = img->getElement(0)[2];

}


void run(MODELTYPE* model){
    using namespace std;
    int size = model->getSize();
    int channels = std::tuple_size<PIXELTYPE>::value;


    for (int i = 0; i < size; i++)
    {
        for (int x = 0; x <channels; x++ )
        {
            int value = img->getElement(i)[x];
            if (value< minMaxRGB[x][0])
            {
                minMaxRGB[x][0] = value;
            }

            if (value> minMaxRGB[x][1])
            {
                minMaxRGB[x][1] = value;
            }
        }
    }

}



};

template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE, class PIXELTYPE>
class StandardContrastStretchingFilter : public opencal::CALLocalFunction<DIMENSION,NEIGHBORHOOD,COORDINATE_TYPE>{

    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    opencal::CALSubstate<PIXELTYPE,DIMENSION>* img;

    double ** minMaxRGB;
public:
    StandardContrastStretchingFilter(auto* sbs , double ** _minMaxRGB):
        img(sbs) , minMaxRGB(_minMaxRGB){

    }


    int computeOutput(int x, int low_in, int low_out, int high_in, int high_out)
    {
        float result;
        if(0 <= x && x <= low_in){
            result = low_out/low_in * x;
        }else if(low_in < x && x <= high_in){
            result = ((high_out - low_out)/(high_in - low_in)) * (x - low_in) + low_out;
        }else if(high_in < x && x <= 255){
            result = ((255 - high_out)/(255 - high_in)) * (x - high_in) + high_out;
        }
        return (int)result;
    }

    int computeValOutput (double x, double low_in, double high_in, double low_out, double high_out)
    {
        if (x <= low_in)
            return (int)low_out;
        else if (x >= high_in)
            return (int)high_out;
        else
        {
            return (int) (low_out + ((x-low_in)* ((high_out-low_out)/(high_in-low_in))));
        }

    }

    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,DIMENSION>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);



        for (int i=0; i<channels; ++i)
        {
            //            int output = computeOutput(newVal[i], minMaxRGB[i][0], 0,minMaxRGB[i][1] , 255);
            int output = computeValOutput(newVal[i],minMaxRGB[i][0], minMaxRGB[i][1], 0,255);
            newVal[i] = cv::saturate_cast<uchar>(output);
        }

        img->setElement(indices,newVal);
    }



};



#endif
