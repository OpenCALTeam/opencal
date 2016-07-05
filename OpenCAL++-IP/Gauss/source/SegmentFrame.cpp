
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
typedef opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2,MOORERADIUS>,COORD_TYPE >> CALRUN;




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

    LabelConnectedComponentFilter(decltype(binImg) sbs,decltype(connComponents)connComp, std::vector<shared_ptr<Bacterium>>* _paths): binImg(sbs), connComponents(connComp),paths(_paths) {
        label = 1;

    }

    void dfs(MODELTYPE* model,std::array<uint,2>& indices, uint label, shared_ptr<Bacterium> & bacterium) {
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
    void setLabel (uint _label)
    {
        label = _label;
    }


    void run(MODELTYPE* model,std::array<uint,2>& indices) {
        using namespace std;
        PIXELTYPE val=binImg->getElement(indices);
        if(val[0]) {
            uint comp = connComponents->getElement(indices);
            if(!comp) {
                shared_ptr<Bacterium> bacterium (new Bacterium());
                dfs(model,indices,label,bacterium);
                label++;

//                std::cout<<" label è "<<label<<" "<<paths->size()<<endl;
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

    friend std::ostream& operator<<(std::ostream& out, Frame f)
    {
        for (int i = 0; i < f.matrix.size(); ++i) {
            for (int j = 0; j < f.matrix[i].size(); ++j) {
                if (f.matrix[i][j] == -1)
                    out<<"  ";
                else
                    out<<f.matrix[i][j]<<" ";
            }
            out<<"\n";

        }
        return out;
    }

};

std::array<COORD_TYPE, 2> coords = { 431,512 };
opencal::CALMooreNeighborhood<2,MOORERADIUS> neighbor;
void SegmentFrame(const std::string& path, Frame& frame, MODELTYPE& calmodel, CALRUN& calrun) {


    //size of the images to br processed


    opencal::CALSubstate<vec1s, 2, COORD_TYPE> *bgr = calmodel.addSubstate<vec1s>();
    opencal::CALSubstate<uint, 2, COORD_TYPE> *connComp = calmodel.addSingleLayerSubstate<uint>();

    std::cout<<"SONO ARRIVATO QUA"<<std::endl;
    //Image Filters
    ContrastStretchingFilter <2,decltype(neighbor),COORD_TYPE,vec1s>contrastStretchingFilter(bgr, 0, 1799, 0, 65535,0.10);

    ThresholdFilter<2,decltype(neighbor),COORD_TYPE,vec1s> thresholdFilter (bgr,0,61680,0,65535);

    RemoveSinglePixelFilter<vec1s> removeSinglePixelFilter(bgr);

    LabelConnectedComponentFilter<vec1s> connComponent(bgr,connComp,&(frame.segmented_bacteria));



    //load image into the model
    bgr->loadSubstate(*(new std::function<decltype(loadImage<vec1s>)>(loadImage<vec1s>)), path);

    calmodel.addElementaryProcess(&contrastStretchingFilter);
    calmodel.addElementaryProcess(&thresholdFilter);
    calmodel.addElementaryProcess(&removeSinglePixelFilter);
    calmodel.addElementaryProcess(&connComponent);


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
    {
        for( auto p : b->points)
            frame.matrix[p.x()][p.y()] = c;
        c++;
    }


    delete connComp;

}
std::set <int> findBacteria (CGALPoint & centroid, Frame & frame, int ray)
{
    std::set <int> bacteria;

    int nCol = frame.matrix[0].size(), nRows = frame.matrix.size();

    int xMin = centroid.x()-ray>=0? centroid.x()-ray:0, xMax = centroid.x()+ray<=nCol? centroid.x()+ray: nCol;
    int yMin = centroid.y()-ray>=0? centroid.y()-ray:0, yMax = centroid.y()+ray<=nRows? centroid.y()+ray: nRows;

    for(int i = xMin; i< xMax; ++i)
    {
        for (int j = yMin; j < yMax; ++j) {
            if (frame.matrix[i][j] != -1)
            {
                bacteria.insert(frame.matrix[i][j]);
            }

        }
    }
    return bacteria;
}

std::set<shared_ptr<Bacterium>> getListBacteria (Frame & frame, std::set <int> & bacteriaIndexes)
{
    std::set<shared_ptr<Bacterium>> bacteria;
    std::set <int>:: iterator it;
    for (it =bacteriaIndexes.begin(); it != bacteriaIndexes.end(); it++) {
        bacteria.insert(frame.segmented_bacteria[*it]);
    }
}

int computeWeight (Bacterium & b1, Bacterium & b2) //TODO k * distance + k1 * area
{
    return 0,6*b1.distance(b2) + 0,4 * (std::abs (b1.getArea()-b2.getArea()));
}

int computeRadius (Bacterium & bacterium) //TODO polygon "radius" + costant
{
    return bacterium.getRadius()+5;
}

void findAnotherCandidate (int i, Frame & frame, std::vector<int>& assignedBacteriaFrame,
                           std::vector <std::list<std::pair <int, int> > >& weights );

void assign (int i, Frame & frame, std::vector<int>& assignedBacteriaFrame,
             std::vector <std::list<std::pair <int, int> > >& weights )

{
    int indexCadidate = weights[i].front().first;
    if (assignedBacteriaFrame[indexCadidate] == -1)
    {
        //            bacteria[i].push_back (frame.segmented_bacteria[indexCadidate]);
        assignedBacteriaFrame[indexCadidate] = i;
        //        assigned[indexCadidate] = true;
        //        numberOfAssigned++;
    }
    else
        findAnotherCandidate (i, frame, /*assigned, numberOfAssigned, */assignedBacteriaFrame, weights);
}

void printWeights (Frame & frame, std::vector <std::list<std::pair <int, int> > >& weights, std::vector <std::list<shared_ptr<Bacterium>> > & bacteria)
{
    for (int i = 0; i < weights.size(); i++)
    {

        std::cout<<" ____________________________"<<std::endl;
        std::cout<< "batterio n "<<i<<" lista condivisa = "<< bacteria[i].back()->getCentroid()<<std::endl;
        for (auto it = weights[i].begin(); it !=weights[i].end(); it++  )
        {
            std::cout<<"peso = "<<it->second<< " batterio frame = "<<frame.segmented_bacteria[it->first]->getCentroid()<<std::endl;
        }
        std::cout<<" ____________________________"<<std::endl;

    }
}

void findAnotherCandidate (int i, Frame & frame, std::vector<int>& assignedBacteriaFrame,
                           std::vector <std::list<std::pair <int, int> > >& weights )
{
    if (weights[i].size() == 0) //untraceable bacterium
    {
        return;
    }

    std::pair <int,int> candidate = weights[i].front(); //coppia id, peso del batterio che voglio assegnare al batterio i della lista condivisa

    int indexOldAssociated = assignedBacteriaFrame[candidate.first]; // indice (nella lista condivisa) del batterio a cui era associato precedentemente
    int oldWeight = weights[indexOldAssociated].front().second; // peso del batterio (nella lista condivisa) a cui era associato precedentemente

    if (candidate.second <= oldWeight) // significa che il batterio era già stato assegnato al suo corrispondente e bisogna cercare di associarlo al successivo
    {
        //prova ad assegnare al secondo candidato
        weights[i].pop_front ();
        //assegna al primo disponibile nella lista dei weights
        assign(i,frame, assignedBacteriaFrame, weights);
    }
    else
    {
        assignedBacteriaFrame[candidate.first] = i; //associo al batterio del frame il nuovo batterio della lista condivisa che meglio matcha con esso
        weights[indexOldAssociated].pop_front (); //tolgo il miglior candidato del batterio (nella lista condivisa) che non matcha più con il primo della lista
        findAnotherCandidate(indexOldAssociated, frame, assignedBacteriaFrame, weights);
    }

}

void tracking (Frame & frame, std::vector <std::list<shared_ptr<Bacterium>> > & bacteria)
{
    std::vector <std::list<std::pair <int, int> > > weights; //std::pair <position in frame.segmented_bacteria vector, distance from bacterium in shared list>

    for (int i = 0; i < bacteria.size(); i++)
    {
        std::set <int> neighbors = findBacteria(bacteria[i].back()->getCentroid(), frame, computeRadius (*bacteria[i].back())); //fix ray

        weights.push_back(std::list<std::pair<int,int>> ());
        std::set <int>:: iterator it;
        for (it =neighbors.begin(); it != neighbors.end(); it++) {
            int weight = computeWeight(*bacteria[i].back(), *(frame.segmented_bacteria[*it]));
            weights[i].push_back (std::pair <int, int> (*it, weight));
        }
        weights[i].sort([](auto &left, auto &right) {
            return left.second < right.second;
        });

    }

//    printWeights(frame,weights, bacteria);

    //    std::vector <bool> assigned (frame.segmented_bacteria.size(), false);
    std::vector <int> assignedBacteriaFrame (frame.segmented_bacteria.size(), -1);

    //    int numberOfAssigned = 0;

    std::cout<<"i batteri sono "<<bacteria.size()<< " i pesi sono "<<weights.size()<<std::endl;
    for (int i = 0; i < bacteria.size(); i++)
    {
        if (weights[i].size() == 0) //untraceable bacterium
        {
            continue;
        }
        assign(i,frame,/*assigned,numberOfAssigned,*/ assignedBacteriaFrame, weights);
    }


    for(int i= 0; i < frame.segmented_bacteria.size(); ++i)
    {
        if (assignedBacteriaFrame[i] == -1)
        {
            std::list <shared_ptr<Bacterium>> l;
            l.push_back(frame.segmented_bacteria[i]);
            bacteria.push_back(l);
        }
        else
        {
            bacteria[assignedBacteriaFrame[i]].push_back (frame.segmented_bacteria[i]);
        }
    }

    //se ci sono batteri nuovi aggiungili alla lista condvisa
    //    if (numberOfAssigned < frame.segmented_bacteria.size())
    //        for (int i = 0; i < frame.segmented_bacteria.size(); i++)
    //        {
    //            if (assignedBacteriaFrame[i] == -1)
    //            {
    //                std::list <shared_ptr<Bacterium>> l;
    //                l.push_back(frame.segmented_bacteria[i]);
    //                bacteria.push_back(l);
    //            }
    //        }
}

// void tracka (list batteri trackati, frame)
//scorri la lista dei batteri
//pesa distanza e area per ogni singolo batterio con gli quelli del frame

//lista tutti possibili candidati
//scegli il candidato migliore

//oppure
//scegli un candidato e in caso cambialo se migliore pretendente



//batterio candidato + peso batterio candidato
//per ogni batterio quello con mimina distanza
//situazione limite batterio senza candidati = batterio sparito
//situazione limite batterio nuovo add alla lista




int main() {
    std::vector <std::list<shared_ptr<Bacterium>> > bacteria;
    std::array<std::string,2> paths= {"./input/tiff/traking_10x_480010persect0001.tif",
                                      "./input/tiff/traking_10x_480010persect0001.tif"};






    MODELTYPE calmodel (
                coords,
                &neighbor,
                opencal::calCommon::CAL_SPACE_TOROIDAL,
                opencal::calCommon::CAL_NO_OPT);


    const int steps  = 1;

    //image processing kernels and filters
    CALRUN calrun (&calmodel, 1, steps, opencal::calCommon::CAL_UPDATE_IMPLICIT);

    for(auto& p : paths){
        std::cout<<"carico "<<p<<"\n";
        Frame f;
        SegmentFrame(p,f, calmodel, calrun);
//        std::cout<<f;
        tracking(f,bacteria);
        std::cout<<"ho finito di segmentare il frame "<<std::endl;

        calmodel.empty();



    }
    //    SegmentFrame(paths[0],f, calmodel, calrun);

    int count= 0;
    for (int i = 0; i < bacteria.size(); i++)
    {
        std::cout<<"batterio "<<i <<" suoi associati "<<bacteria[i].size()<<"\n";

        if (bacteria[i].size() >1)
        {
            count++;
        }
//        for (auto b : bacteria[i])
//        {
//            std::cout<<*b<< " ";
//        }
    }

    std::cout<<"i batteri associati sono "<<count<<std::endl;




    return 0;
}



