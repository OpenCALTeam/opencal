
#include <OpenCAL++/calElementaryProcessFunctor.h>


CALElementaryProcessFunctor::CALElementaryProcessFunctor() {

}

CALElementaryProcessFunctor::~CALElementaryProcessFunctor() {

}
 void CALElementaryProcessFunctor::operator()(CALModel* calModel, int* indexes)
{
    this->run(calModel,indexes);
}

