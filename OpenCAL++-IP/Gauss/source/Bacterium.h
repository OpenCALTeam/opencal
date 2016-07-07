#ifndef BACTERIUM_H
#define BACTERIUM_H

#include <CGAL/Boolean_set_operations_2.h>
//#include <CGAL/convex_hull_2.h>
#include<set>
#include <CGAL/centroid.h>

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes;
typedef CGAL::Polygon_2<K> CGALPolygon;

typedef K::Point_2 CGALPoint;
typedef std::set<CGALPoint> Points;


class Polygon : public CGALPolygon
{

private:
    CGALPoint centroid;
    int area;
    int radius;



public:
    Polygon (Points points) : CGALPolygon (points.begin(), points.end())
    {
        //        CGAL::convex_hull_2( points.begin(), points.end(), std::back_inserter(*this) );
        create();

    }

    Polygon (CGALPoint* firstPoint, CGALPoint* lastPoint) : CGALPolygon (firstPoint, lastPoint)
    {
        create();

    }

    Polygon()
    {

    }

    void create ()
    {
        area = this->size();
        centroid=CGAL::centroid(this->vertices_begin(), this->vertices_end(),CGAL::Dimension_tag<0>());
        computeRadius();
    }

    int getArea ()
    {
        return area;
    }

    int getRadius ()
    {
        return radius;
    }

    void convexHull ()
    {
        //        CGAL::convex_hull_2(this->vertices_begin(), this->vertices_end(), std::back_inserter(*this) );
        //        this->erase(this->vertices_begin(), this->vertices_begin()+area);
    }


    bool check_inside(CGALPoint& pt)
    {
        CGAL::Bounded_side bounded_side =this->bounded_side(pt);
        return bounded_side == CGAL::ON_BOUNDED_SIDE  || bounded_side == CGAL::ON_BOUNDARY;
    }

    int intersectionArea (Polygon & polygon)
    {
        Polygon::iterator it;

        int area = 0;
        for (it = polygon.vertices_begin(); it != polygon.vertices_end(); it++) {

            if (check_inside(*it))
                area++;
        }
        return area;
    }

    //    int differenceArea (Polygon polygon)
    //    {
    //        return 0;
    //    }

    CGALPoint& getCentroid ()
    {
        return centroid;

    }

    Points computePixels ()
    {
        Points pixelsToColor;
        Points points;
        typedef CGALPolygon::iterator LIT;
        for(LIT lit = this->vertices_begin(); lit!=this->vertices_end()-1; lit++){
            points = computePixelsSegment(*lit, *(lit+1));
            pixelsToColor.insert(points.begin(), points.end());
        }
        points = computePixelsSegment(*this->vertices_begin(), *(this->vertices_end()-1));
        pixelsToColor.insert(points.begin(), points.end());
        return pixelsToColor;
    }

    friend std::ostream& operator<<(std::ostream& os, Polygon polygon)
    {
        os << "Centroid: " << polygon.centroid<<" \n"<<"Area: "<<polygon.area<<"\n"<<"Radius: "<<polygon.radius<<"\n"<<"Points: ";
        CGALPolygon::iterator it;
        for(it = polygon.vertices_begin(); it!=polygon.vertices_end(); it++){
            os<<*it<<" | ";
        }
        os<<"\n";
        return os;
    }



private:

    Points computePixelsSegment (CGALPoint start, CGALPoint end)
    {

        Points pixelsToColor;
        start = CGALPoint ((int)start.x(), (int)start.y());
        end = CGALPoint ((int)end.x(), (int)end.y());

        int dx =  abs(end.x()-start.x()), sx = start.x()<end.x() ? 1 : -1;
        int dy = -abs(end.y()-start.y()), sy = start.y()<end.y() ? 1 : -1;
        int err = dx+dy, e2; /* error value e_xy */


        int x = start.x(), y = start.y();
        for(;;){  /* loop */

            pixelsToColor.insert(CGALPoint (x,y));
            if (x==(int)end.x() && y==(int)end.y())
                break;
            e2 = 2*err;
            if (e2 >= dy)
            {
                err += dy;
                x += sx;
            } /* e_xy+e_x > 0 */
            if (e2 <= dx)
            {
                err += dx;
                y += sy;
            } /* e_xy+e_y < 0 */
        }
        return pixelsToColor;
    }


    void computeRadius ()
    {
        Polygon::iterator it;
        radius = 0;

        int distanceFromCentroid;
        for (it = this->vertices_begin(); it != this->vertices_end(); it++) {
            distanceFromCentroid = std::sqrt (std::pow ((it->x() - this->centroid.x()), 2) + std::pow ((it->y()- this->centroid.y()),2));
            if (distanceFromCentroid > radius)
                radius = distanceFromCentroid;
        }

    }



};


class Bacterium
{
    //private:
public:
    Polygon polygon;
    Points points;
    bool lost = false;
    Bacterium(){}

    /*    Bacterium (Points _points) : points(_points) , polygon(points)
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


    int getIntersectionArea (Bacterium&  bacterium)
    {
        return polygon.intersectionArea(bacterium.polygon);
    }

    double distance (Bacterium& bacterium)
    {
        return std::sqrt (std::pow ((bacterium.getCentroid().x() - this->polygon.getCentroid().x()), 2) + std::pow ((bacterium.getCentroid().y()- this->polygon.getCentroid().y()),2));
    }

    int getRadius ()
    {
        return polygon.getRadius();
    }

    void createBactriaFromRawPoints(){
        Polygon p(points);
        polygon = p;

    }





    friend std::ostream& operator<<(std::ostream& os, Bacterium b)
    {
        os << b.polygon<<" \n";
        return os;
    }
};

#endif
