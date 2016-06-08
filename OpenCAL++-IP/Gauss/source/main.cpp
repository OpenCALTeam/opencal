#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include<OpenCAL++/functional_utilities.h>
#include "image_processing.h"
#include<algorithm>
#include "ContrastStretchingFilter.h"
typedef unsigned int COORD_TYPE;

using namespace std::placeholders;

constexpr unsigned int MOORERADIUS=1;


typedef std::array<unsigned char, 2> vec2b;
typedef std::array<unsigned char, 3> vec3b;
typedef std::array<unsigned char, 4> vec4b;
typedef std::array<unsigned short, 1> vec1s;

typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2,MOORERADIUS>, COORD_TYPE> MODELTYPE;




std::array<COORD_TYPE, 2> coords = { 512,512 };
CALLBACKTYPE<vec1s>::SAVECALLBACK savef = std::bind(save<vec1s>,_1,_2,coords[0], coords[1],CV_16U);


template<typename PIXELTYPE>
class BrightnessContrastFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint>{

    opencal::CALSubstate<PIXELTYPE,2>* img;
    double alpha;
    double beta;
public:
    BrightnessContrastFilter(auto* sbs , double _alpha, double _beta): img(sbs) , alpha(_alpha), beta(_beta){

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);

        unsigned short ns = model->getNeighborhoodSize();
        for (int i=0; i<channels; ++i)
            newVal[i] = min(255.0, max(alpha*newVal[i]+beta, 0.0));

        img->setElement(indices,newVal);
    }



};



template<typename PIXELTYPE>
class MeanFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint>{

    opencal::CALSubstate<PIXELTYPE,2>* img;
public:
    MeanFilter(auto* sbs): img(sbs){

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices){
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        std::array<unsigned int,channels> avg ={};
        PIXELTYPE newval=img->getElement(indices);

        unsigned short ns = model->getNeighborhoodSize();
        for(int x=0 ; x<ns; ++x){
            for (int i=0; i<channels; ++i)
                avg[i] += img->getX(indices,x)[i];
        }
        // cout<<(unsigned short)avg[0]<<" ";
        //cout<<endl;

        auto f = [&](unsigned int c)-> unsigned int{return ns!=0 ? c/ns : c;};
        opencal::map_inplace(avg.begin(),avg.end(),f);
        for(int i=0; i<channels; i++)
            newval[i] = avg[i];

        img->setElement(indices,newval);
    }



};




template<typename PIXELTYPE>
class SaveGlobalFunction : public opencal::CALGlobalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint>{

    opencal::CALSubstate<PIXELTYPE,2>* img;
    int step = 0;
    std::string filenameExtension;
public:
    SaveGlobalFunction(auto* sbs,  enum opencal::calCommon :: CALUpdateMode _UPDATE_MODE, std::string _filenameExtension): CALGlobalFunction(_UPDATE_MODE), img(sbs), filenameExtension(_filenameExtension){

    }


    void run(MODELTYPE* model){
        img->saveSubstate(savef, "output/out_"+std::to_string(step)+"."+filenameExtension);
        step++;
    }

};

template<uint _DIMENSION, class _NEIGHBORHOOD , class _KERNEL , class _SUBSTATE , class COORDINATE_TYPE = uint>
class UniformFilter : public opencal::ConvolutionFilter<_DIMENSION, _NEIGHBORHOOD , _KERNEL, _SUBSTATE >{
    typedef opencal::ConvolutionFilter<_DIMENSION, _NEIGHBORHOOD , _KERNEL, _SUBSTATE > SUPER;

    //constructor inheritance
    using SUPER::SUPER;

    virtual void applyConvolution(typename SUPER::MODEL_pointer model, std::array<uint,_DIMENSION>& indices, _KERNEL* kernel) {
        typename _SUBSTATE::PAYLOAD newVal= {} ;

        for(int i=0 ; i < model->getNeighborhoodSize() ; ++i)
            for(int j=0 ; j < newVal.size(); j++)
                newVal[j] += std::get<0>((*kernel)[i]) * (double)this->substate->getX(indices,i)[j];

        this->substate->setElement(indices,newVal);

    }

};



int main ()
{


    int steps=1; printf("how many steps?.."); scanf("%d",&steps);

    opencal::UniformKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> unikernel;
    opencal::GaussianKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> gaukernel
            ({0.5,0.5},{0.0,0.0});

    opencal::LaplacianKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> lapkernel
            ({0.3,0.3},{0.0,0.0});



    // unikernel.print();
    // gaukernel.print();
    //     lapkernel.print();
    opencal::CALMooreNeighborhood<2,MOORERADIUS> neighbor;




    //return 0;

    MODELTYPE calmodel(
                coords,
                &neighbor,
                opencal::calCommon::CAL_SPACE_TOROIDAL,
                opencal::calCommon::CAL_NO_OPT);

    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2,MOORERADIUS>,
            COORD_TYPE >> calrun(&calmodel, 1, steps, opencal::calCommon::CAL_UPDATE_IMPLICIT);

    opencal::CALSubstate<vec1s, 2, COORD_TYPE> *bgr = calmodel.addSubstate<vec1s>();

    //laplacian
    //UniformFilter<2, decltype(neighbor), decltype(lapkernel),opencal::CALSubstate<vec3b, 2, COORD_TYPE>  > unifilter(bgr,&lapkernel);
    //UniformFilter<2, decltype(neighbor), decltype(gaukernel),opencal::CALSubstate<vec1s, 2, COORD_TYPE>  > unifilter(bgr,&gaukernel);

    ContrastStretchingFilter <2,decltype(neighbor),COORD_TYPE,vec1s>* contrastStretchingFilter = new ContrastStretchingFilter<2,decltype(neighbor),COORD_TYPE,vec1s> (bgr, 0, 1799, 0, 65535,0.10);

    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec1s>)>(loadImage<vec1s>)), "input/tiff/traking_10x_480010persect0001.tif");



    calmodel.addElementaryProcess(contrastStretchingFilter);

    calmodel.addElementaryProcess(new SaveGlobalFunction<vec1s>(bgr, opencal::calCommon::CAL_UPDATE_EXPLICIT, "tif"));
    calrun.run();
    //bgr->saveSubstate(savef, "output/outproteinMO.jpg");

    printf("END\n");


}
