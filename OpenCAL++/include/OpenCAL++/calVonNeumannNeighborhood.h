//

// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
#define OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H

namespace opencal {
template<unsigned int DIMENSION>
class CALVonNeumannNeighborhood {
protected:
static constexpr const int total = 2*DIMENSION+1;
typedef std::array<uint,DIMENSION> element;

  static std::array<element, total> indices;

   void defineNeighborhood();

public:

  CALVonNeumannNeighborhood() {
    defineNeighborhood();
  }

  static const auto&  getNeighborhoodIndices() {
    return indices;
  }
};

template<>
inline void opencal::CALVonNeumannNeighborhood<2>::defineNeighborhood(){
    CALVonNeumannNeighborhood<2>::indices = { {
                                                { {1, 2} },
                                                { { 4, 5} },
                                                { {1, 2} },
                                                { {1, 2} },
                                                { {1, 2} }
                                            } };
}


template<>
inline void opencal::CALVonNeumannNeighborhood<3>::defineNeighborhood(){
    CALVonNeumannNeighborhood<3>::indices =   { {
                                                { {1, 2,3} },
                                                { { 4, 5,3} },
                                                { {1, 2,3} },
                                                { {1, 2,3} },
                                                { {1, 2,3} }
                                            } };

}




// static member declaration
template<uint DIMENSION>
std::array<typename CALVonNeumannNeighborhood<DIMENSION>::element, CALVonNeumannNeighborhood<DIMENSION>::total>opencal::CALVonNeumannNeighborhood<DIMENSION>::indices;

} // namespace opencal

#endif // OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
