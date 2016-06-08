#ifndef INVERTCOLORSFILTER_H
#define INVERTCOLORSFILTER_H

#include <OpenCAL++/calElementaryProcessFunctor.h>
#include<OpenCAL++/calSubstate.h>


template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE, class PIXELTYPE>
class ThresholdFilter : public opencal::CALLocalFunction< DIMENSION , NEIGHBORHOOD , COORDINATE_TYPE>{

    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    typedef typename std::tuple_element<0,PIXELTYPE>::type TYPE;
    opencal::CALSubstate<PIXELTYPE,DIMENSION>* img;
    TYPE lowT , highT ;
    TYPE outRangeValue , inRangeValue;
public:
    ThresholdFilter(auto* sbs, TYPE _lowT , TYPE _highT , TYPE _outrange, TYPE _inrange ):
        img(sbs), lowT(_lowT) , highT(_highT) , outRangeValue(_outrange) , inRangeValue(_inrange) { }



    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,DIMENSION>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);

        for (int i = 0; i<channels; ++i)
        {
            const TYPE output =  (newVal[i] >= lowT && newVal[i] <= highT) ? inRangeValue : outRangeValue ;
            newVal[i] = output;
        }

        img->setElement(indices,newVal);
    }


};



#endif

