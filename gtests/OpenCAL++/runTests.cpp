//
// Created by Davide Spataro on 12/04/16.
//



#include <gtest/gtest.h>
#include "functional_tests.h"
#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calModel.h>
#include <OpenCAL++/CALVonNeumannNeighborhood.h>

int main(int argc, char** argv){
    typedef uint COORD_TYPE;

    std::array<COORD_TYPE,2> coords  = {10,10};
    opencal::CALVonNeumannNeighborhood neighbor;
    opencal::CALModel<2,opencal::CALVonNeumannNeighborhood,COORD_TYPE> calmodel(coords,&neighbor , opencal::calCommon::CAL_SPACE_FLAT , opencal::calCommon::CAL_NO_OPT );
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    return 0;
}

