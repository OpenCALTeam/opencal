#include "CALManagerImageIO.h"
#include <cctype>
#include <clocale>
#include <OpenCAL++/calCommon.h>


int main(int argc, char** argv)
{
    std::array<uint, 2> coord = {480,640};
    int size = opencal::calCommon::multiplier<2,uint>(coord, 0, 2);
    cv::Vec4b* array = opencal::CALImageManagerIO<2>::
            loadBuffer<cv::Vec4b>(size, (char*)"alpha.png");

//    for (int i = 0; i < size; ++i) {
//        std::cout<<array[i] <<"   ";
//    }

    opencal::CALImageManagerIO<2>::saveBuffer<cv::Vec4b>(array,size,coord, "prova.png");

    delete[] array;



    return 0;

}

