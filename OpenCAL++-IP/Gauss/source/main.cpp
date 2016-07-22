#include "ParticleTracking.h"
#include "LabelConnectedComponentFilter.h"
#include "MyMat.h"


typedef std::array<unsigned short, 1> vec1s;
using namespace std;
constexpr unsigned int MOORERADIUS=1;
typedef unsigned int COORD_TYPE;
typedef opencal::CALModel<2, opencal::CALMooreNeighborhood<2,MOORERADIUS>, COORD_TYPE> MODELTYPE;
typedef opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2,MOORERADIUS>,COORD_TYPE >> CALRUN;
typedef opencal::CALSubstate<vec1s, 2, COORD_TYPE> CALSUBSTATE;

class MyRun : public opencal::CALRun<MODELTYPE>
{
public:

    MyRun (CALMODEL_pointer calModel,   int _initial_step,int _final_step,enum opencal::calCommon :: CALUpdateMode _UPDATE_MODE
           )	:
        CALRun(calModel, _initial_step,_final_step, _UPDATE_MODE)
    {

    }

    virtual void finalize()
    {
        calModel->resetProcesses();
    }


};



int main ()
{

    //traking_10x_480010persect
    std::array<COORD_TYPE, 2> coords = { 431,512 };

    //100_0019t
    //std::array<COORD_TYPE, 2> coords = { 402,512 };


    opencal::CALMooreNeighborhood<2,MOORERADIUS> neighbor;
    MODELTYPE calmodel (
                coords,
                &neighbor,
                opencal::calCommon::CAL_SPACE_TOROIDAL,
                opencal::calCommon::CAL_NO_OPT);
    const int steps  = 1;
    std::string path = "/home/parcuri/Dropbox/Workspace_cpp/OpenCAL_Devel/opencal/OpenCAL++-IP/Gauss/input/tiff/traking_10x_480010persect";
    CALSUBSTATE* bgr = calmodel.addSubstate<vec1s>();
    MyRun calrun (&calmodel, 1, steps, opencal::calCommon::CAL_UPDATE_IMPLICIT);
    ContrastStretchingFilter <2,decltype(neighbor),COORD_TYPE,vec1s>contrastStretchingFilter(bgr, 0, 1799, 0, 65535,0.10);
    ThresholdFilter<2,decltype(neighbor),COORD_TYPE,vec1s> thresholdFilter (bgr,0,61680,0,65535);

    //    RemoveSinglePixelFilter<vec1s> removeSinglePixelFilter(bgr);

    Frame<2,COORD_TYPE>* frame = new Frame<2,COORD_TYPE>();

    opencal::CALSubstate<uint, 2, COORD_TYPE> *connComp = calmodel.addSingleLayerSubstate<uint>();
    LabelConnectedComponentFilter<vec1s, decltype(neighbor), COORD_TYPE> connComponent(bgr,connComp,&(frame->segmented_particles));

    calmodel.addElementaryProcess(&contrastStretchingFilter);
    calmodel.addElementaryProcess(&thresholdFilter);
    //    calmodel.addElementaryProcess(&removeSinglePixelFilter);
    calmodel.addElementaryProcess(&connComponent);


    ParticlesTracking<2,decltype(neighbor),COORD_TYPE> particlesTracking (&calmodel, &calrun, frame);
    particlesTracking.execute<vec1s> (path,bgr, 2100, 4);


    auto particles =particlesTracking.getParticles();
    std::cout<<"ci sono "<<particles.size()<<" batteri"<<std::endl;

    MyMat mat (coords[0], coords[1], CV_8UC3);
    mat.addBacteria(particles);
    mat.saveImage("bact.png");
    delete frame;

    return 0;
}
