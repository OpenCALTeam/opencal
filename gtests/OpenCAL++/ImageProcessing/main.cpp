#include <cctype>
#include <clocale>
#include <OpenCAL++/calCommon.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include<vector>

#include <opencv2/opencv.hpp>


template<class T>
T* load (std::string pathInput)
{
    cv::Mat mat= cv::imread(pathInput, CV_LOAD_IMAGE_UNCHANGED);

//        std::cout<<mat.channels()<<std::endl;
    printf ("%d %d \n", mat.rows, mat.cols);

    T* array = new T [mat.rows* mat.cols];
    int linearIndex = 0;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
            T& bgra = mat.at<T>(i, j);
            array[linearIndex] = bgra;
        }
    }

    return array;


}

template<class T>
void save (T *array, int rows, int cols, int type, std::string pathOutput)
{


    cv::Mat mat (rows, cols, type);
    int linearIndex =0;
    printf ("%d %d \n", mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
            mat.at<T>(i,j) = array[linearIndex];

        }

    }
    cv::imwrite(pathOutput, mat);
    return;


}


template<class T>
void display (T *buffer,int rows, int cols, int type)
{
    cv::Mat mat (rows, cols, type);
    int linearIndex =0;
    printf ("%d %d \n", mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
            mat.at<T>(i,j) = buffer[linearIndex];

        }

    }

    cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", mat );
    cv::waitKey(0);


}

int main(int argc, char** argv)
{

    cv::Vec4b* loadPng = load<cv::Vec4b>("image/png/alpha.png");
    save<cv::Vec4b>(loadPng, 480,640,CV_8UC4,"image/png/prova.png");

    cv::Vec3b* loadTiff = load<cv::Vec3b>("image/tiff/example2.tiff");
    save<cv::Vec3b>(loadTiff, 512,640,CV_8UC3,"image/tiff/example2Stampa.tiff");

    cv::Vec3b* loadJpg = load<cv::Vec3b>("image/jpg/image.jpg");
    save<cv::Vec3b>(loadJpg, 711, 1024,CV_8UC3,"image/jpg/out.jpg");



    display(loadTiff,512,640, CV_8UC3);

    delete [] loadJpg;
    delete [] loadTiff;
    delete [] loadPng;

    return 0;

}

