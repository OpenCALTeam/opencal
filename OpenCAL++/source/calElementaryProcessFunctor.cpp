<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calElementaryProcessFunctor.h>
=======
#include <OpenCAL++11/calElementaryProcessFunctor.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calElementaryProcessFunctor.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

CALElementaryProcessFunctor::CALElementaryProcessFunctor() {

}

CALElementaryProcessFunctor::~CALElementaryProcessFunctor() {

}
 void CALElementaryProcessFunctor::operator()(CALModel* calModel, int* indexes)
{
    this->run(calModel,indexes);
}

