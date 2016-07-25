#ifndef PARTICLESTRACKING_H
#define PARTICLESTRACKING_H

#include "ContrastStretchingFilter.h"
#include "ThresholdFilter.h"
#include "image_processing.h"

#include <sys/stat.h> //stat function
#include <list>
#include<vector>

#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include<OpenCAL++/calRun.h>
#include "Particle.h"



template < uint DIMENSION, typename COORDINATE_TYPE = uint>
class Frame
{
public:

    std::vector<std::shared_ptr<Particle>> segmented_particles;
    std::vector<std::vector<int>> matrix;

    Frame(){}
    Frame(const std::vector<std::shared_ptr<Particle>>& segmented_particles,
          std::array <COORDINATE_TYPE, DIMENSION>& coordinates)
    {
        this->segmented_particles = segmented_particles;
        init(coordinates);

    }

    void init (std::array <COORDINATE_TYPE, DIMENSION>& coordinates)
    {
        for(auto b : this->segmented_particles) {
            b->createBactriaFromRawPoints();
        }



        for(int i= 0 ; i < coordinates[0] ; i++) {
            std::vector<int> row(coordinates[1],-1);
            this->matrix.push_back(row);
        }

        int c=0;
        for(auto b : this->segmented_particles)
        {
            for( auto p : b->points)
                this->matrix[p.x()][p.y()] = c;
            c++;
        }
    }

    std::set <int> findParticles (CGALPoint & centroid, int radius)
    {
        std::set <int> particles;

        int nCol = this->matrix[0].size(), nRows = this->matrix.size();

        int xMin = centroid.x()-radius>=0? centroid.x()-radius:0, xMax = centroid.x()+radius<=nRows? centroid.x()+radius: nRows;
        int yMin = centroid.y()-radius>=0? centroid.y()-radius:0, yMax = centroid.y()+radius<=nCol? centroid.y()+radius: nCol;

        for(int i = xMin; i< xMax; ++i)
        {
            for (int j = yMin; j < yMax; ++j) {
                if (this->matrix[i][j] != -1)
                {
                    particles.insert(this->matrix[i][j]);
                }

            }
        }
        return particles;
    }


    std::set<std::shared_ptr<Particle>> getListParticle (std::set <int> & particlesIndexes)
    {
        std::set<std::shared_ptr<Particle>> particles;
        std::set <int>:: iterator it;
        for (it =particlesIndexes.begin(); it != particlesIndexes.end(); it++) {
            particles.insert(this->segmented_particles[*it]);
        }
    }

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

    void clear ()
    {
        matrix.clear();
        segmented_particles.clear();
    }

};




template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
class ParticlesTracking
{
private:
    typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> MODELTYPE;
    typedef Frame<DIMENSION,COORDINATE_TYPE> FRAME;
    typedef opencal::CALRun <MODELTYPE> CALRUN;
    MODELTYPE* calmodel;
    CALRUN* calrun;
    std::vector <std::list<std::shared_ptr<Particle>> > particles;

    FRAME* frame;



    std::string ToString(int value,int digitsCount)
    {
        using namespace std;
        ostringstream os;
        os<<setfill('0')<<setw(digitsCount)<<value;
        return os.str();
    }
    void assign (int i,std::vector<int>& assignedParticlesFrame,
                 std::vector <std::list<std::pair <int, int> > >& weights)

    {

        if (weights[i].size() != 0)
        {

            int indexCadidate = weights[i].front().first;
            if (assignedParticlesFrame[indexCadidate] == -1)
            {
                assignedParticlesFrame[indexCadidate] = i;
                particles[i].back()->lost=0;
            }
            else
                findAnotherCandidate (i, assignedParticlesFrame, weights);
        }
        else //untraceable bacterium
            particles[i].back()->lost ++;
    }

    void findAnotherCandidate (int i, std::vector<int>& assignedParticlesFrame,
                               std::vector <std::list<std::pair <int, int> > >& weights)
    {
        if (weights[i].size() == 0) //untraceable bacterium
        {
            particles[i].back()->lost ++;
            return;
        }

        // std::cout<<"cerco un altro candidato per "<<i << " la size dei candidati è "<< weights[i].size()<<std::endl;

        std::pair <int,int> candidate = weights[i].front(); //coppia id, peso del batterio che voglio assegnare al batterio i della lista condivisa

        int indexOldAssociated = assignedParticlesFrame[candidate.first]; // indice (nella lista condivisa) del batterio a cui era associato precedentemente
        int oldWeight = weights[indexOldAssociated].front().second; // peso del batterio (nella lista condivisa) a cui era associato precedentemente

        if (candidate.second >= oldWeight) // significa che il batterio era già stato assegnato al suo corrispondente e bisogna cercare di associarlo al successivo
        {

            //prova ad assegnare al secondo candidato
            weights[i].pop_front ();
            //assegna al primo disponibile nella lista dei weights
            assign(i, assignedParticlesFrame, weights);
        }
        else
        {
            assignedParticlesFrame[candidate.first] = i; //associo al batterio del frame il nuovo batterio della lista condivisa che meglio matcha con esso
            weights[indexOldAssociated].pop_front (); //tolgo il miglior candidato del batterio (nella lista condivisa) che non matcha più con il primo della lista
            assign(indexOldAssociated, assignedParticlesFrame, weights);
        }

    }


    void tracking ()
    {
        std::vector <std::list<std::pair <int, int> > > weights; //std::pair <position in frame.segmented_bacteria vector, distance from bacterium in shared list>

        for (int i = 0; i < particles.size(); i++)
        {
            std::set <int> neighbors = frame->findParticles(particles[i].back()->getCentroid(), computeRadius (*particles[i].back())); //fix ray

            weights.push_back(std::list<std::pair<int,int>> ());
            std::set <int>:: iterator it;
            for (it =neighbors.begin(); it != neighbors.end(); it++) {
                int weight = computeWeight(*particles[i].back(), *(frame->segmented_particles[*it]));
                weights[i].push_back (std::pair <int, int> (*it, weight));
            }
            weights[i].sort([](auto &left, auto &right) {
                return left.second < right.second;
            });

        }


        std::vector <int> assignedParticlesFrame (frame->segmented_particles.size(), -1);

        for (int i = 0; i < particles.size(); i++)
        {
            if (particles[i].back()->lost<5) //TODO mettere parametro da settare
            {
                // std::cout<<" sono qui a processare il batterio n "<<i<<std::endl;

                assign(i, assignedParticlesFrame, weights);
            }
        }


        for(int i= 0; i < frame->segmented_particles.size(); ++i)
        {
            if (assignedParticlesFrame[i] == -1)
            {
                std::list <std::shared_ptr<Particle>> l;
                l.push_back(frame->segmented_particles[i]);
                particles.push_back(l);
            }
            else
            {
                particles[assignedParticlesFrame[i]].push_back (frame->segmented_particles[i]);
            }
        }

    }
public:

    ParticlesTracking (MODELTYPE* _calmodel, CALRUN* _calrun, FRAME * _frame)
    {
        this->calmodel = _calmodel;
        this->frame = _frame;
        this->calrun = _calrun;
    }


    ParticlesTracking (FRAME* frame)
    {
        this->frame =frame;
    }


    std::vector <std::list<std::shared_ptr<Particle>> >& getParticles ()
    {
        return particles;
    }

    virtual int computeWeight (Particle & p1, Particle & p2) //TODO k * distance + k1 * area
    {
        int res= /*0.6**/p1.distance(p2) /*+ 0.4 * (std::abs (b1.getArea()-b2.getArea()))*/;

        return res;
    }

    virtual int computeRadius (Particle & particle) //TODO polygon "radius" + costant
    {
        return 10;//bacterium.getRadius();
    }


    template <typename PIXELTYPE>
    void execute (std::string path,opencal::CALSubstate<PIXELTYPE, DIMENSION, COORDINATE_TYPE> * bgr, int numberOfFrame, int digits)
    {


        for(int i = 1; i<= numberOfFrame; i++) {
            std::string currentPath = path+ToString(i,digits)+".tif";
            std::cout<<"sono current path "<<currentPath<<std::endl;
            bgr->loadSubstate(*(new std::function<decltype(loadImage<PIXELTYPE>)>(loadImage<PIXELTYPE>)), currentPath);
            calrun->run();
            frame->init(calmodel->getCoordinates());
            tracking();
            std::cout<<"ho processato "<<frame->segmented_particles.size()<<" particelle"<<std::endl;
            frame->clear();

        }
    }

};

#endif
