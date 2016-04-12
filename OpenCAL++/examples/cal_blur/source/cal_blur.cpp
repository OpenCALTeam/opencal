#include <cstdlib>
#include <iostream>
#include <cmath>
#include <memory>

#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calSubstate.h>
#include <OpenCAL++/calRun.h>


class CABlur {
  unsigned int ROWS, COLS;

  CALSubstate<unsigned char> *red;
  CALSubstate<unsigned char> *green;
  CALSubstate<unsigned char> *blue;
  CALSubstate<unsigned char> *alpha;

  CALModel *blur2D;
  CALRun   *blur2Dsimulation;


 };

int main() {
  std::cout << "END" << ::std::endl;

  return 0;
}
