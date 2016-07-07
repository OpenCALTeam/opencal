#ifndef MYMAT_H
#define MYMAT_H

#include <opencv2/opencv.hpp>
#include <memory>
using namespace cv;
using namespace std;
#include"Bacterium.h"
class MyMat : public Mat
{

private:
    void create()
    {

        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                Vec3b& bgra = this->at<Vec3b>(i, j);
                bgra[0] = saturate_cast<uchar>(0); // Blue
                bgra[1] = saturate_cast<uchar>(0); // Green
                bgra[2] = saturate_cast<uchar>(0); // Red

            }
        }
    }

    void addBacterium (Points & pixelsToColor, int r, int g, int b)
    {
        for (Points::iterator it= pixelsToColor.begin(); it!= pixelsToColor.end(); it++)
        {
            Vec3b& bgra = this->at<Vec3b>(it->x(), it->y());
            bgra[0] = saturate_cast<uchar>(b); // Blue
            bgra[1] = saturate_cast<uchar>(g); // Green
            bgra[2] = saturate_cast<uchar>(r); // Red


        }
    }

    Points computePixels (std::list<shared_ptr<Bacterium>> & bacterium)
    {
        Points pixelsToColor;
        Points points;

        for(auto lit = bacterium.begin(); lit!=std::prev(bacterium.end()); lit++){
            points = computePixelsSegment((*lit)->getCentroid(), (*std::next(lit, 1))->getCentroid());
            pixelsToColor.insert(points.begin(), points.end());
        }

        //        points = computePixelsSegment((*bacterium.begin())->getCentroid(), (*std::prev(bacterium.end()))->getCentroid());
        //        pixelsToColor.insert(points.begin(), points.end());
        return pixelsToColor;
    }

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


public:
    MyMat(int rows, int cols, int type): Mat (rows, cols, type) {
        create();
    }


    void addBacteria (std::vector <std::list<shared_ptr<Bacterium>> > & bacteria)
    {

        for (int i= 0; i < bacteria.size(); i++)
        {
            if (bacteria[i].size() > 0)
            {

                Points points;
                for (auto it = bacteria[i].begin(); it != bacteria[i].end(); it++)
                {
                    points.insert((*it)->getCentroid());
                }
                addBacterium(points, std::rand()%255, std::rand()%255, std::rand()%255);
            }
        }
    }




    int saveImage (std::string nameFile)
    {


        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        try {
            imwrite(nameFile, *this, compression_params);
        }
        catch (std::runtime_error& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved PNG file with alpha data.\n");
        return 0;
    }


};




#endif
