//

// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
#define OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H

namespace opencal {
template<unsigned int DIMENSION>
class CALVonNeumannNeighborhood {
protected:

  static std::array<uint, DIMENSION> indices;
  void defineNeighborhood() {
    int n = calModel->getDimension();

    assert(n == 2 || n == 3);

    if (n == 2) {
      /*
       | 1 |
          ---|---|---
           2 | 0 | 3
          ---|---|---
       | 4 |
       */
      int vonNeumannNeighborhoodIndexes[5][2] = { {  0,   0 },
                                                  { -1,   0 },
                                                  {  0,  -1 },
                                                  {  0,   1 },
                                                  {  1,   0 } };

      for (int i = 0; i < 5; i++) {
        calModel->addNeighbor(vonNeumannNeighborhoodIndexes[i]);
      }
    }
    else {
      /*
           slice -1       slice 0       slice 1
           (sopra)					  (sotto)
       |   |         | 1 |         |   |
          ---|---|---   ---|---|---   ---|---|---
       | 5 |       2 | 0 | 3       | 6 |
          ---|---|---   ---|---|---   ---|---|---
       |   |         | 4 |         |   |
       */
      int vonNeumannNeighborhoodIndexes[7][3] = { {  0,   0,   0 },
                                                  { -1,   0,   0 },
                                                  {  0,  -1,   0 },
                                                  {  0,   1,   0 },
                                                  {  1,   0,   0 },
                                                  {  0,   0,  -1 },
                                                  {  0,   0,   1 } };

      for (int i = 0; i < 7; i++) {
        calModel->addNeighbor(vonNeumannNeighborhoodIndexes[i]);
      }
    }
  }

public:

  CALVonNeumannNeighborhood() {
    defineNeighborhood();
  }

  static const auto&  getNeighborhoodIndices() {
    return indices;
  }
};


// static member declaration
template<uint DIMENSION>
std::array<uint, DIMENSION>opencal::CALVonNeumannNeighborhood<DIMENSION>::indices;
} // namespace opencal
#endif // OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
