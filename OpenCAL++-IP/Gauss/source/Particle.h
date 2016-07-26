#ifndef PARTICLE_H
#define PARTICLE_H

#include "Polygon.h"
class Particle
{
private:
public:
    Polygon polygon;
    Points points;
    int lost = 0;
    int frame;
    Particle(){}

    /*    Particle (Points _points) : points(_points) , polygon(points)
    {

    }*/


    CGALPoint& getCentroid ()
    {
        return polygon.getCentroid();
    }

    int getArea ()
    {
        return polygon.getArea();
    }


    int getIntersectionArea (Particle&  particle)
    {
        return polygon.intersectionArea(particle.polygon);
    }

    double distance (Particle& particle)
    {
        return std::sqrt (std::pow ((particle.getCentroid().x() - this->polygon.getCentroid().x()), 2) + std::pow ((particle.getCentroid().y()- this->polygon.getCentroid().y()),2));
    }

    int getRadius ()
    {
        return polygon.getRadius();
    }

    void createBactriaFromRawPoints(){
        Polygon p(points);
        polygon = p;

    }





    friend std::ostream& operator<<(std::ostream& os, Particle b)
    {
        os << b.polygon <<" " <<b.frame<<" \n";
        return os;
    }
};
#endif
