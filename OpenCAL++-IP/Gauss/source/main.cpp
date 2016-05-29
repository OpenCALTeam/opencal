#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include<OpenCAL++/functional_utilities.h>
typedef unsigned int COORD_TYPE;

using namespace std::placeholders;

constexpr unsigned int MOORERADIUS=5;


typedef std::array<unsigned char, 2> vec2b;
typedef std::array<unsigned char, 3> vec3b;
typedef std::array<unsigned char, 4> vec4b;

typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2,MOORERADIUS>, COORD_TYPE> MODELTYPE;

template<class T>
T *loadImage(int size, const std::string& path){
//    printf("sto qui\n");
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
void save (const T *array, const std::string pathOutput,int rows, int cols, int type)
{


    cv::Mat mat (rows, cols, type);
    int linearIndex =0;
    //printf ("%d %d \n", mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j, ++linearIndex)
            mat.at<T>(i,j) = array[linearIndex];
    }
    cv::imwrite(pathOutput, mat);
    return;
}

template<typename T>
class CALLBACKTYPE{
public:
    typedef std::function<void(const T*, const std::string&)> SAVECALLBACK;
    typedef std::function<T*(int size, const std::string&)>    LOADCALLBACK;

};

std::array<COORD_TYPE, 2> coords = { 857,1500 };
CALLBACKTYPE<vec3b>::SAVECALLBACK savef = std::bind(save<vec3b>,_1,_2,coords[0], coords[1],CV_8UC3);

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
public:
    SaveGlobalFunction(auto* sbs,  enum opencal::calCommon :: CALUpdateMode _UPDATE_MODE): CALGlobalFunction(_UPDATE_MODE), img(sbs){

    }


    void run(MODELTYPE* model){
        img->saveSubstate(savef, "output/out_"+std::to_string(step)+".jpg");
        step++;
    }

};

int main ()
{


    int steps=1; printf("how many steps?.."); scanf("%d",&steps);


    opencal::CALMooreNeighborhood<2,MOORERADIUS> neighbor;

    MODELTYPE calmodel(
                coords,
                &neighbor,
                opencal::calCommon::CAL_SPACE_TOROIDAL,
                opencal::calCommon::CAL_NO_OPT);

    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2,MOORERADIUS>,
            COORD_TYPE >> calrun(&calmodel, 1, steps, opencal::calCommon::CAL_UPDATE_IMPLICIT);

    opencal::CALSubstate<vec3b, 2, COORD_TYPE> *bgr = calmodel.addSubstate<vec3b>();

    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec3b>)>(loadImage<vec3b>)), "input/jpg/protein1500.jpg");


    calmodel.addElementaryProcess(new MeanFilter<vec3b>(bgr));
    calmodel.addElementaryProcess(new SaveGlobalFunction<vec3b>(bgr, opencal::calCommon::CAL_UPDATE_EXPLICIT));
    calrun.run();
  //  bgr->saveSubstate(savef, "output/outproteinMO.jpg");




}
