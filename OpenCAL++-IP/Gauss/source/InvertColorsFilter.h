#ifndef INVERTCOLORSFILTER_H
#define INVERTCOLORSFILTER_H


#include <OpenCAL++/calElementaryProcessFunctor.h>

#include<OpenCAL++/calSubstate.h>
#include<opencv2/core/core.hpp>



template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE, class PIXELTYPE>
class InvertColorsFilter : public opencal::CALLocalFunction<DIMENSION,NEIGHBORHOOD,COORDINATE_TYPE>{

    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    opencal::CALSubstate<PIXELTYPE,DIMENSION>* img;
    int maxLimit;

public:
    InvertColorsFilter(auto* sbs, int _maxLimit=255 ):
        img(sbs), maxLimit(_maxLimit)  {

    }



    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,DIMENSION>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);


        for (int i=0; i<channels; ++i)
        {
            int output = maxLimit - newVal[i];
            newVal[i] = cv::saturate_cast<uchar>(output);
        }

        img->setElement(indices,newVal);
    }



};




#endif

