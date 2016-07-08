#include <iostream>
#include <fstream>
#include  <cstdlib>
#include <string>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_2 CGALPoint;

#include <opencv2/opencv.hpp>
#include <memory>
using namespace cv;
using namespace std;


class Bact
{
public:
    CGALPoint point;
    Bact (int x,int y) : point(x,y)
    {

    }

    void move ()
    {
        point = CGALPoint (((int)point.x()+(rand()%5-2))%431,((int)point.y()+(rand()%5-2))%512);
    }

};

class NewMat : public Mat
{

private:
    void create()
    {

        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                Vec2b& bgra = this->at<Vec2b>(i, j);
                bgra[0] = saturate_cast<uchar>(0);
                bgra[1] = saturate_cast<uchar>(0);

            }
        }
    }

    void addBacterium (CGALPoint & pixelsToColor, int r, int g, int b)
    {

            Vec2b& bgra = this->at<Vec2b>(pixelsToColor.x(), pixelsToColor.y());
            bgra[0] = saturate_cast<uchar>(255);
            bgra[1] = saturate_cast<uchar>(255);

    }




public:
    NewMat(int rows, int cols, int type): Mat (rows, cols, type) {
        create();
    }


    void addBacteria (std::vector <Bact > & bacteria)
    {


        for (int i = 0; i < bacteria.size(); i++ )
        {

                addBacterium(bacteria[i].point, 255, 255, 255);

        }
    }




    int saveImage (std::string nameFile)
    {

        try {
            imwrite(nameFile, *this);
        }
        catch (std::runtime_error& ex) {
            fprintf(stderr, "Exception converting image to TIF format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved TIF file with alpha data.\n");
        return 0;
    }


};







void update (std::vector<Bact>& bacteria)
{
    for(int i = 0; i< bacteria.size(); i++)
    {
        bacteria[i].move();
    }
}


std::string ToString(int value,int digitsCount)
{
  using namespace std;
    ostringstream os;
    os<<setfill('0')<<setw(digitsCount)<<value;
    return os.str();
}


void bacteriaGenerator ()
{
    std::vector<Bact> bacteria;
    for (int i= 0; i< 20; i++)
    {
        bacteria.push_back(Bact(rand()%431, rand()%512));
    }

    for (int i= 0; i< 35; i++)
    {
        NewMat mat (431,512,CV_16U);
        mat.addBacteria(bacteria);
        mat.saveImage("./input/generated/bacteria"+ToString(i,2)+".tif");
        update(bacteria);
    }

}

//int main()
//{

//    bacteriaGenerator();



//    return 0;
//}






















