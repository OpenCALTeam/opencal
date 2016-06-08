#ifndef CONTRASTSTRETCHINGFILTER_H
#define CONTRASTSTRETCHINGFILTER_H


#include <OpenCAL++/calElementaryProcessFunctor.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calSubstate.h>
#include<opencv2/core/core.hpp>



template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE, class PIXELTYPE>
class ContrastStretchingFilter : public opencal::CALLocalFunction<DIMENSION,NEIGHBORHOOD,COORDINATE_TYPE>{

    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    opencal::CALSubstate<PIXELTYPE,DIMENSION>* img;
    typedef  typename std::tuple_element<0,PIXELTYPE>::type TYPE;

    TYPE low_out;
    TYPE low_in;
    TYPE high_out;
    TYPE high_in;
    double gamma;
public:
    ContrastStretchingFilter(auto* sbs , TYPE _low_in, TYPE _high_in, TYPE _low_out, TYPE _high_out, double _gamma=1):
        img(sbs) , low_out(_low_out), low_in(_low_in), high_out(_high_out), high_in(_high_in), gamma (_gamma) {

    }


    TYPE computeOutput(TYPE x)
    {
        TYPE result;
        if(0 <= x && x <= low_in){
            result = low_out/low_in * x;
        }else if(low_in < x && x <= high_in){
            result = ((high_out - low_out)/(high_in - low_in)) * (x - low_in) + low_out;
        }else if(high_in < x && x <= 255){
            result = ((255 - high_out)/(255 - high_in)) * (x - high_in) + high_out;
        }
        return result;
    }

    TYPE computeValOutput (TYPE x)
    {
        if (x <= low_in)
            return low_out;
        else if (x >= high_in)
            return high_out;
        else
        {
            return (low_out + ((x-low_in)* ((high_out-low_out)/(high_in-low_in))));
        }

    }
    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,DIMENSION>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);


        double gammaCorrection = 1.0 / gamma;

        for (int i=0; i<channels; ++i)
        {
            TYPE output = computeValOutput(newVal[i]);
            output = (double)high_out * std::pow(((double)output  / (double)high_out), gammaCorrection);

            newVal[i] = cv::saturate_cast<TYPE>(output);
        }

        img->setElement(indices,newVal);
    }



};




#endif
