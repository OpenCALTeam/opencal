#ifndef OPENCAL_ALL_CALIMAGEMANAGERIO_H
#define OPENCAL_ALL_CALIMAGEMANAGERIO_H


#include <OpenCAL++/calCommon.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include<vector>
#pragma once

#include <opencv2/opencv.hpp>
namespace opencal {

template<uint DIMENSION, typename COORDINATE_TYPE = uint>
class CALImageManagerIO {
public:

    /*! \brief loads a matrix from file.
        */
    template<class T, class STR_TYPE = std::string>
    static T *loadBuffer(int size, const STR_TYPE& path){

        cv::Mat mat= cv::imread(path,-1);

        T* vec = new T [size];
        int linearIndex = 0;

//        std::cout<<mat<<std::endl;

        std::cout<<mat.channels()<<std::endl;

        printf ("%d %d \n", mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
                T& bgra = mat.at<T>(i, j);
                vec[linearIndex] = bgra;
            }
        }


        return vec;
    }


    template<class T, class STR_TYPE = std::string>
    inline static T *loadBuffer(std::array <COORDINATE_TYPE, DIMENSION>& coordinates, const STR_TYPE& path){
        return loadBuffer(opencal::calCommon::multiplier(coordinates, 0, DIMENSION),path);
    }

    /*! \brief saves a certain matrix to file.
        */
    template<class T, class STR_TYPE = std::string>
    static void  saveBuffer(T *buffer, int size, std::array <COORDINATE_TYPE, DIMENSION>& coordinates, const STR_TYPE& path) {

        cv::Mat mat (coordinates[0], coordinates[1], CV_8UC4);
        int linearIndex =0;
        printf ("%d %d \n", mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
                mat.at<cv::Vec4b>(i,j) = buffer[linearIndex];

            }

        }

        std::vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
        cv::imwrite(path, mat,compression_params);



    }



};



}//namespace opencal

#endif //OPENCAL_ALL_CALCONVERTERIO_H
