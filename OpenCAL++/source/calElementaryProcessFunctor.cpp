#include <OpenCAL++11/calElementaryProcessFunctor.h>

CALElementaryProcessFunctor::CALElementaryProcessFunctor() {

}

CALElementaryProcessFunctor::~CALElementaryProcessFunctor() {

}
 void CALElementaryProcessFunctor::operator()(CALModel* calModel, int* indexes)
{
    this->run(calModel,indexes);
}

