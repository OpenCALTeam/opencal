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


    int getIntersectionArea (Bacterium&  bacteria)
    {
        return polygon.intersectionArea(bacteria.polygon);
    }




};

#endif
