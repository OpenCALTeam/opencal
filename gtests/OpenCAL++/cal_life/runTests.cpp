//
// Created by Davide Spataro on 12/04/16.
//
#include <gtest/gtest.h>
#include "functional_tests.h"
#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>
#include <OpenCAL++/calRealConverter.h>
typedef unsigned int COORD_TYPE;

class Life_transition_function : public opencal::CALElementaryProcessFunctor<2,
                                                                             opencal::CALMooreNeighborhood<2>
                                                                            >{
private:

  opencal::CALSubstate<int, 2> *Q;

public:

  Life_transition_function(opencal::CALSubstate<int, 2> *_Q)
  {
    Q = _Q;
  }

  void run(opencal::CALModel<2, opencal::CALMooreNeighborhood<2> > *calModel,
           std::array<COORD_TYPE, 2>& indexes)
  {
    int sum              = 0, n;
    int neighborhoodSize = calModel->getNeighborhoodSize();

    for (n = 1; n < neighborhoodSize; n++)
    {
      sum += Q->getX(indexes, n);
    }


    if ((sum == 3) || ((sum == 2) && (Q->getElement(indexes) == 1)))
    {
      Q->setElement(indexes, 1);
    }
    else Q->setElement(indexes, 0);
  }
};




int main(int argc, char **argv) {
  std::array<COORD_TYPE, 2> coords = { 8, 16 };
  opencal::CALMooreNeighborhood<2> neighbor;

  opencal::CALModel<2, opencal::CALMooreNeighborhood<2>, COORD_TYPE> calmodel(
    coords,
    &neighbor,
    opencal::calCommon::CAL_SPACE_TOROIDAL,
    opencal::calCommon::CAL_NO_OPT);

  opencal::CALRun < opencal::CALModel < 2, opencal::CALMooreNeighborhood<2>,
  COORD_TYPE >> calrun(&calmodel, 1, 4, opencal::calCommon::CAL_UPDATE_IMPLICIT);

  opencal::CALSubstate<int, 2, COORD_TYPE> *Q = calmodel.addSubstate<int>();

  calmodel.initSubstate(Q, 0);

  std::array<std::array<COORD_TYPE, 2>, 5> indexes = {        {
                                                                { { 0, 2 } },
                                                                { { 1, 0 } },
                                                                { { 1, 2 } },
                                                                { { 2, 1 } },
                                                                { { 2, 2 } }
                                                              } };

  // set a glider
  for (uint i = 0; i < 5; i++)
  {
    calmodel.init(Q, indexes[i], 1);
  }


  calmodel.addElementaryProcess(new Life_transition_function(Q) );

  Q->getCurrent()->stampa(coords);
  printf("\n_____________________________________ \n\n");
  calrun.run();

  Q->getCurrent()->stampa(coords);

    opencal::CALIntConverter converter;
    Q->saveSubstate<opencal::CALIntConverter> (converter, (char*)"./substate.txt");


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();



    return 0;
}
