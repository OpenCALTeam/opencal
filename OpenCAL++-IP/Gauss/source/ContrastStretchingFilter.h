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

    double low_out;
    double low_in;
    double high_out;
    double high_in;
public:
    ContrastStretchingFilter(auto* sbs , double _low_in, double _high_in, double _low_out, double _high_out):
        img(sbs) , low_out(_low_out), low_in(_low_in), high_out(_high_out), high_in(_high_in) {

    }


    int computeOutput(int x)
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

    int computeValOutput (double x)
    {
        if (x <= low_in)
            return (int)low_out;
        else if (x >= high_in)
            return (int)high_out;
        else
        {
            return (low_out + ((x-low_in)* ((high_out-low_out)/(high_in-low_in))));
        }

    }
    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,DIMENSION>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);


        for (int i=0; i<channels; ++i)
        {
            int output = computeValOutput(newVal[i]);
            newVal[i] = cv::saturate_cast<uchar>(output);
        }

        img->setElement(indices,newVal);
    }



};




#endif
