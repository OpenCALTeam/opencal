#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calModel.h>
#include<OpenCAL++/calRun.h>
#include <OpenCAL++/calMooreNeighborhood.h>

#define S_X (100)
#define S_y (100)
#define S_Z (100)
#define S_T (100)


constexpr int DIMENSION = 4;
constexpr int MOORERADIUS = 1;

typedef unsigned int COORD_TYPE;
typedef  opencal::CALMooreNeighborhood<4,MOORERADIUS> NEIGHBORHOOD;
typedef opencal::CALModel<DIMENSION, NEIGHBORHOOD, COORD_TYPE> MODELTYPE;

std::array<COORD_TYPE, DIMENSION> coords = { S_X, S_Y,S_Z,S_T };
int main(){



  return 0;

}

