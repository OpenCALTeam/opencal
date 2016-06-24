#ifndef POLYGON_H
#define POLYGON_H
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

    }

    int getArea ()
    {
        return area;
    }


    void convexHull ()
    {
        //        CGAL::convex_hull_2(this->vertices_begin(), this->vertices_end(), std::back_inserter(*this) );
        //        this->erase(this->vertices_begin(), this->vertices_begin()+area);
    }


    CGAL::Bounded_side check_inside(CGALPoint& pt)
    {
        return this->bounded_side(pt);
    }

    int intersectionArea (Polygon & polygon)
    {
       Polygon::iterator it;

       int area = 0;
       for (it = polygon.vertices_begin(); it != polygon.vertices_end(); it++) {
            CGAL::Bounded_side bounded_side = check_inside(*it);
            if (bounded_side == CGAL::ON_BOUNDED_SIDE  || bounded_side == CGAL::ON_BOUNDARY)
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




};




#endif
