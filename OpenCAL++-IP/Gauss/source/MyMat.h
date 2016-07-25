#ifndef MYMAT_H
#define MYMAT_H

#include <opencv2/opencv.hpp>
#include <memory>
using namespace cv;
using namespace std;
#include"Particle.h"


class Colour {
public:
    std::array<uint,4> cols {{255,255,255,255}};

};

class MyMat : public Mat
{

private:
    static vector<Colour> sharedCols;
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

    Points computePixels (std::list<shared_ptr<Particle>> & bacterium)
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

    void initColors ()
    {
        for(int k =0 ; k < 1000; k++) {
            Colour c;
            int r = rand()%150+1;
            c.cols[0] = r + rand()%104;
            c.cols[1] = r + rand()%104;
            c.cols[2] =  r + rand()%104;
            MyMat::sharedCols.push_back(c);
        }
    }


public:
    MyMat(int rows, int cols, int type): Mat (rows, cols, type) {
        create();
        if (MyMat::sharedCols.empty())
        {
            initColors();
        }
    }


    void addBacteria (std::vector <std::list<shared_ptr<Particle>> > & bacteria)
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
                int size = MyMat::sharedCols.size();
                addBacterium(points,MyMat::sharedCols[i%size].cols[0], MyMat::sharedCols[i%size].cols[1] ,MyMat::sharedCols[i%size].cols[2]);
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



vector<Colour> MyMat::sharedCols;


#endif
