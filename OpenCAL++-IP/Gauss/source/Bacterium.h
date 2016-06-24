#ifndef BACTERIUM_H
#define BACTERIUM_H
#include"Polygon.h"


class Bacterium
{
private:
Polygon polygon;

public:

     Bacterium (Points points) : polygon (points)
     {

     }

    CGALPoint& getCentroid ()
    {
        polygon.getCentroid();
    }

    int getArea ()
    {
        return polygon.getArea();
    }


    int getIntersectionArea (Bacterium&  bacterium)
    {
        return polygon.intersectionArea(bacterium.polygon);
    }

    double distance (Bacterium& bacterium)
    {
        return std::sqrt (std::pow ((bacterium.getCentroid().x() - this->polygon.getCentroid().x()), 2) + std::pow ((bacterium.getCentroid().y()- this->polygon.getCentroid().y()),2));

    }




};

#endif
