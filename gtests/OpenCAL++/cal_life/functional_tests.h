//
// Created by Davide Spataro on 12/04/16.
//

#ifndef OPENCAL_ALL_FUNCTIONAL_TESTS_H
#define OPENCAL_ALL_FUNCTIONAL_TESTS_

#include <gtest/gtest.h>
#include <OpenCAL++/calCommon.h>



TEST(functional_utilities, fold_multiplier){

    std::array<uint , 1> b = {123};

    std::array<uint , 3> c = {0,2,3};

    std::array<uint , 4> d = {1,2,3,4};

    uint mfold=opencal::calCommon::multiplier<1,uint>(b,0);
    ASSERT_EQ(mfold,123);

    mfold=opencal::calCommon::multiplier<3,uint>(c,0);
    ASSERT_EQ(mfold,0);

    mfold=opencal::calCommon::multiplier<4,uint>(d,0);
    ASSERT_EQ(mfold,24);


}


#endif //OPENCAL_ALL_FUNCTIONAL_TESTS_H
