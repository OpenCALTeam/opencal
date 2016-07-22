#ifndef LABELCONNECTEDCOMPONETFILTER_H
#define LABELCONNECTEDCOMPONETFILTER_H


#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include "Particle.h"


template<typename PIXELTYPE, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
class LabelConnectedComponentFilter : public opencal::CALLocalFunction<2,NEIGHBORHOOD,COORDINATE_TYPE> {

    typedef opencal::CALModel<2, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    opencal::CALSubstate<PIXELTYPE,2>* binImg;
    opencal::CALSubstate<COORDINATE_TYPE,2>*     connComponents;
    uint label;
public:
    std::vector<std::shared_ptr<Particle>>* paths;

    LabelConnectedComponentFilter(decltype(binImg) sbs,decltype(connComponents)connComp, std::vector<std::shared_ptr<Particle>>* _paths): binImg(sbs), connComponents(connComp),paths(_paths) {
        label = 1;
        connComponents->setCurrentBuffer(0);

    }

    void dfs(MODELTYPE* model,std::array<COORDINATE_TYPE,2>& indices, uint label, std::shared_ptr<Particle> & particle) {
        PIXELTYPE val = binImg->getElement(indices);

        if(val[0] && !connComponents->getElement(indices)) {
            connComponents->setElementCurrent(indices,label);

            CGALPoint p(indices[0],indices[1]);
            particle->points.insert(p);


            for(int i=1 ; i<model->getNeighborhoodSize() ; i++) {
                uint linearIndex = opencal::calCommon::cellLinearIndex<2,uint>(indices,model->getCoordinates());
                int linearIndexN = opencal::CALNeighborPool<2,uint>::getNeighborN(linearIndex,i);
                auto indices_x =opencal::calCommon::cellMultidimensionalIndices<2,uint>(linearIndexN);
                dfs(model,indices_x,label,particle);
            }

        }

    }
    void setLabel (uint _label)
    {
        label = _label;
    }


    void reset ()
    {
        label = 1;
        connComponents->setCurrentBuffer(0);
    }

    void run(MODELTYPE* model,std::array<COORDINATE_TYPE,2>& indices) {
        using namespace std;
        PIXELTYPE val=binImg->getElement(indices);
        if(val[0]) {
            uint comp = connComponents->getElement(indices);
            if(!comp) {
                std::shared_ptr<Particle> particle (new Particle());
                dfs(model,indices,label,particle);
                label++;

                paths->push_back(particle);

            }
        }
    }
}; 

#endif
