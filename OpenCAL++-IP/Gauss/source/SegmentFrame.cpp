
#include <sys/stat.h> //stat function
#include <list>
#include<vector>

#include"Bacterium.h"
#include "ContrastStretchingFilter.h"
#include "ThresholdFilter.h"
#include "image_processing.h"

#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>


#include <opencv2/opencv.hpp>

//utility functions
inline bool file_exists (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

//-------------------------------------
using namespace std;



typedef std::array<unsigned char, 2> vec2b;
typedef std::array<unsigned char, 3> vec3b;
typedef std::array<unsigned char, 4> vec4b;
typedef std::array<unsigned short, 1> vec1s;




//radius of the moor neighboood
constexpr unsigned int MOORERADIUS=1;
typedef unsigned int COORD_TYPE;
typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2,MOORERADIUS>, COORD_TYPE> MODELTYPE;




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
class LabelConnectedComponentFilter : public opencal::CALLocalFunction<2,opencal::CALMooreNeighborhood<2,MOORERADIUS>,uint> {

    opencal::CALSubstate<PIXELTYPE,2>* binImg;
    opencal::CALSubstate<uint,2>*     connComponents;
    uint label;
public:
    std::vector<shared_ptr<Bacterium>>* paths;

    LabelConnectedComponentFilter(decltype(binImg) sbs,decltype(connComponents)connComp, std::vector<Bacterium>* _paths): binImg(sbs), connComponents(connComp),paths(_paths) {
        label = 1;

    }

    void dfs(MODELTYPE* model,std::array<uint,2>& indices, uint label, shared_ptr<Bacterium> bacterium) {
        PIXELTYPE val = binImg->getElement(indices);

        if(val[0] && !connComponents->getElement(indices)) {
            connComponents->setElementCurrent(indices,label);

            CGALPoint p(indices[0],indices[1]);
            bacterium->points.insert(p);

            for(int i=1 ; i<model->getNeighborhoodSize() ; i++) {
                uint linearIndex = opencal::calCommon::cellLinearIndex<2,uint>(indices,model->getCoordinates());
                int linearIndexN = opencal::CALNeighborPool<2,uint>::getNeighborN(linearIndex,i);
                auto indices_x =opencal::calCommon::cellMultidimensionalIndices<2,uint>(linearIndexN);

                dfs(model,indices_x,label,bacterium);
            }

        }

    }


    void run(MODELTYPE* model,std::array<uint,2>& indices) {
        using namespace std;
        PIXELTYPE val=binImg->getElement(indices);
        if(val[0]) {
            uint comp = connComponents->getElement(indices);
            if(!comp) {
                shared_ptr<Bacterium> bacterium;
                dfs(model,indices,label,bacterium);
                label++;
                //std::cout<<label<<" "<<bacteriaPath.size()<<endl;
                paths->push_back(bacterium);

            }
        }
    }
};


class Frame
{
public:

    std::vector<shared_ptr<Bacterium>> segmented_bacteria;
    vector<vector<int>> matrix;

};

void SegmentFrame(const std::string& path, Frame& frame) {

    const int steps  = 1;

    //size of the images to br processed
    std::array<COORD_TYPE, 2> coords = { 431,512 };

    //Moore Neighborhood
    opencal::CALMooreNeighborhood<2,MOORERADIUS> neighbor;

    MODELTYPE calmodel(
        coords,
        &neighbor,
        opencal::calCommon::CAL_SPACE_TOROIDAL,
        opencal::calCommon::CAL_NO_OPT);

//image processing kernels and filters
    opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2,MOORERADIUS>,
            COORD_TYPE >> calrun(&calmodel, 1, steps, opencal::calCommon::CAL_UPDATE_IMPLICIT);

    opencal::CALSubstate<vec1s, 2, COORD_TYPE> *bgr = calmodel.addSubstate<vec1s>();
    opencal::CALSubstate<uint, 2, COORD_TYPE> *connComp = calmodel.addSingleLayerSubstate<uint>();

    //Image Filters
    ContrastStretchingFilter <2,decltype(neighbor),COORD_TYPE,vec1s>contrastStretchingFilter(bgr, 0, 1799, 0, 65535,0.10);

    ThresholdFilter<2,decltype(neighbor),COORD_TYPE,vec1s> thresholdFilter (bgr,0,61680,0,65535);

    RemoveSinglePixelFilter<vec1s> removeSinglePixelFilter(bgr);

   // LabelConnectedComponentFilter<vec1s> connComponent(bgr,connComp,&(frame.segmented_bacteria));



//load image into the model
    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec1s>)>(loadImage<vec1s>)), path);

    calmodel.addElementaryProcess(&contrastStretchingFilter);
    calmodel.addElementaryProcess(&thresholdFilter);
    calmodel.addElementaryProcess(&removeSinglePixelFilter);
    //calmodel.addElementaryProcess(&connComponent);


     calrun.run();

     //frame.segmented has the list of all bacteria each with a list of points
     //Postprocess the bacteria in order to generate the polygon and the ocnvexhull
     for(auto b : frame.segmented_bacteria){
       b->createBactriaFromRawPoints();
     }



    for(int i= 0 ; i < coords[0] ; i++){
      vector<int> row(coords[1],-1);
      frame.matrix.push_back(row);
       }

    int c=0;
    for(auto b : frame.segmented_bacteria)
        for( auto p : b->points)
          frame.matrix[p.x()][p.y()] = c++;




}


int main() {

    return 0;
}



