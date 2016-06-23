#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include<OpenCAL++/calSubstateUnsafe.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include<OpenCAL++/functional_utilities.h>
#include "image_processing.h"
#include<algorithm>
#include "ContrastStretchingFilter.h"
#include "ThresholdFilter.h"
#include "Bacterium.h"
typedef unsigned int COORD_TYPE;

using namespace std::placeholders;

constexpr unsigned int MOORERADIUS=1;


typedef std::array<unsigned char, 2> vec2b;
typedef std::array<unsigned char, 3> vec3b;
typedef std::array<unsigned char, 4> vec4b;
typedef std::array<unsigned short, 1> vec1s;

typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2,MOORERADIUS>, COORD_TYPE> MODELTYPE;
//mogrify -crop 512x431+0+25 traking_10x_480010persec/traking_10x_480010persect00* converted/conv_*



std::array<COORD_TYPE, 2> coords = { 431,512 };
CALLBACKTYPE<vec1s>::SAVECALLBACK savef = std::bind(save<vec1s>,_1,_2,coords[0], coords[1],CV_16U);


template<typename PIXELTYPE>
class BrightnessContrastFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint> {

    opencal::CALSubstate<PIXELTYPE,2>* img;
    double alpha;
    double beta;
public:
    BrightnessContrastFilter(auto* sbs , double _alpha, double _beta): img(sbs) , alpha(_alpha), beta(_beta) {

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices) {
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
class RemoveSinglePixelFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint> {

    opencal::CALSubstate<PIXELTYPE,2>* img;
public:
    RemoveSinglePixelFilter(auto* sbs): img(sbs) {

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices) {
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        PIXELTYPE newVal=img->getElement(indices);
        int count = 0;
        unsigned short ns = model->getNeighborhoodSize();
        for (int i=0; i<ns; ++i) {
            if(img->getX(indices,i)[0])
               count++;

        }

        if(count <= 1)
            for(int i=0; i<channels; i++)
                newVal[i] = 0;

        img->setElement(indices,newVal);

    }



};



template<typename PIXELTYPE>
class MeanFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint> {

    opencal::CALSubstate<PIXELTYPE,2>* img;
public:
    MeanFilter(auto* sbs): img(sbs) {

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices) {
        using namespace std;
        constexpr int channels = std::tuple_size<PIXELTYPE>::value;
        std::array<unsigned int,channels> avg = {};
        PIXELTYPE newval=img->getElement(indices);

        unsigned short ns = model->getNeighborhoodSize();
        for(int x=0 ; x<ns; ++x) {
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
class SaveGlobalFunction : public opencal::CALGlobalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint> {

    opencal::CALSubstate<PIXELTYPE,2>* img;
    int step = 0;
    std::string filenameExtension;
public:
    SaveGlobalFunction(auto* sbs,  enum opencal::calCommon :: CALUpdateMode _UPDATE_MODE, std::string _filenameExtension): CALGlobalFunction(_UPDATE_MODE), img(sbs), filenameExtension(_filenameExtension) {

    }


    void run(MODELTYPE* model) {
        img->saveSubstate(savef, "output/out_"+std::to_string(step)+"."+filenameExtension);
        step++;
    }

};

template<uint _DIMENSION, class _NEIGHBORHOOD , class _KERNEL , class _SUBSTATE , class COORDINATE_TYPE = uint>
class UniformFilter : public opencal::ConvolutionFilter<_DIMENSION, _NEIGHBORHOOD , _KERNEL, _SUBSTATE > {
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







template<typename PIXELTYPE>
class LabelConnectedComponentFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint>{

    opencal::CALSubstate<PIXELTYPE,2>* binImg;
    opencal::CALSubstate<uint,2>*     connComponents;
        uint label;
public:
     std::vector<std::vector<std::array<uint,2>>> paths;

    LabelConnectedComponentFilter(decltype(binImg) sbs,decltype(connComponents)connComp): binImg(sbs), connComponents(connComp),paths(){
      label = 1;

    }


    void dfs(MODELTYPE* model,std::array<uint,2>& indices, uint label,std::vector<std::array<uint,2>>* bacteriaPath){
      PIXELTYPE val = binImg->getElement(indices);

      if(val[0] && !connComponents->getElement(indices)){
           connComponents->setElementCurrent(indices,label);
           bacteriaPath->push_back(indices);

    for(int i=1 ; i<model->getNeighborhoodSize() ; i++){
       uint linearIndex = opencal::calCommon::cellLinearIndex<2,uint>(indices,model->getCoordinates());
       int linearIndexN = opencal::CALNeighborPool<2,uint>::getNeighborN(linearIndex,i);
       auto indices_x =opencal::calCommon::cellMultidimensionalIndices<2,uint>(linearIndexN);

       dfs(model,indices_x,label,bacteriaPath);
    }

      }



    }


    void run(MODELTYPE* model,std::array<uint,2>& indices){
        using namespace std;
        PIXELTYPE val=binImg->getElement(indices);
        if(val[0]){
            uint comp = connComponents->getElement(indices);
            if(!comp){
             std::vector<std::array<uint,2>> bacteriaPath;
             dfs(model,indices,label,&bacteriaPath);
             label++;
             //std::cout<<label<<" "<<bacteriaPath.size()<<endl;
             paths.push_back(bacteriaPath);

            }


        }
    }



};


void saveOutput(auto& paths){
  using namespace std;
 std::ofstream outfile;
outfile.open("trajectory_file.txt");
 outfile<<paths.size()<<endl;
 for(auto path : paths){
  outfile<<path.size()<<" ";
  for(auto coor : path){
    outfile<<coor[0]<<" "<<coor[1] << " ";


  }
  outfile<<endl;

 }

 outfile.close();

}


int main ()
{


    int steps=1;
    printf("how many steps?..");
    scanf("%d",&steps);

    opencal::UniformKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> unikernel;
    opencal::GaussianKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> gaukernel
    ({0.5,0.5}, {0.0,0.0});

    opencal::LaplacianKernel<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>, double> lapkernel
    ({0.3,0.3}, {0.0,0.0});



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
    opencal::CALSubstate<uint, 2, COORD_TYPE> *connComp = calmodel.addSingleLayerSubstate<uint>();

    //laplacian
    //UniformFilter<2, decltype(neighbor), decltype(lapkernel),opencal::CALSubstate<vec3b, 2, COORD_TYPE>  > unifilter(bgr,&lapkernel);
    UniformFilter<2, decltype(neighbor), decltype(gaukernel),opencal::CALSubstate<vec1s, 2, COORD_TYPE>  > unifilter(bgr,&gaukernel);

    ContrastStretchingFilter <2,decltype(neighbor),COORD_TYPE,vec1s>* contrastStretchingFilter = new ContrastStretchingFilter<2,decltype(neighbor),COORD_TYPE,vec1s> (bgr, 0, 1799, 0, 65535,0.10);


    RemoveSinglePixelFilter<vec1s>* removeSinglePixelFilter = new RemoveSinglePixelFilter<vec1s>(bgr);



    LabelConnectedComponentFilter<vec1s>* connComponent = new LabelConnectedComponentFilter<vec1s>(bgr,connComp);
    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec1s>)>(loadImage<vec1s>)), "input/tiff/traking_10x_480010persect0001.tif");



    ThresholdFilter<2,decltype(neighbor),COORD_TYPE,vec1s>* thresholdFilter = new ThresholdFilter<2,decltype(neighbor),COORD_TYPE,vec1s> (bgr,0,61680,0,65535);
    calmodel.addElementaryProcess(contrastStretchingFilter);
    calmodel.addElementaryProcess(thresholdFilter);
    calmodel.addElementaryProcess(removeSinglePixelFilter);
    calmodel.addElementaryProcess(connComponent);

    calmodel.addElementaryProcess(new SaveGlobalFunction<vec1s>(bgr, opencal::calCommon::CAL_UPDATE_EXPLICIT, "tif"));
    calrun.run();
  //  connComp->saveSubstate(opencal::tostring_fn<uint>(), "output/connComp.txt");


   saveOutput(connComponent->paths);


    printf("END\n");


}

/*
 *
 * Blob detection: Laplacian of Gaussian (LoG), Difference of Gaussians (DoG), Determinant of Hessian (DoH), Hough transform.
 * Edge detection: Canny, Deriche, Differential, Sobel, Prewitt, Roberts cross.
 * Corner detection: Harris operator, Shi and Tomasi, Level curve curvature, SUSAN, FAST.
 * Feature description: SIFT, SURF, GLOH, HOG.
 *
 * */
