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
//int is necessary since neighborhood definition requires negative numbers
typedef std::array<int,DIMENSION> element;

  static std::array<element, total> indices;
public:
   static void defineNeighborhood(){
       indices[0] = {0};//central cell
       //total number of insertions is 2*Dimension+1
       for (int i = 0; i < DIMENSION; ++i)
       {
           indices[2*i+1] = {0};
           indices[2*i+2] = {0};
           indices[2*i+1][i] = 1;
           indices[2*i+2][i] = -1;
       }
   }



  CALVonNeumannNeighborhood() {
    defineNeighborhood();
  }

  static const auto&  getNeighborhoodIndices() {

    return indices;
  }
};

/*
template<>
inline void opencal::CALVonNeumannNeighborhood<2>::defineNeighborhood(){
    CALVonNeumannNeighborhood<2>::indices = { {
                                                { {0, 0} },
                                                { { 1, 0} },
                                                { {-1, 0} },
                                                { {0, 1} },
                                                { {0, -1} }
                                            } };
}


template<>
inline void opencal::CALVonNeumannNeighborhood<3>::defineNeighborhood(){
    CALVonNeumannNeighborhood<3>::indices =   { {
                                                { {0, 2,3} },
                                                { {1, 5,3} },
                                                { {-1, 2,3} },
                                                { {0, 5,3} },
                                                { {0, 2,3} },
                                                { {0, 2,3} },
                                                { {0, 2,3} }
                                            } };

}

*/


// static member declaration
template<uint DIMENSION>
std::array<typename CALVonNeumannNeighborhood<DIMENSION>::element, CALVonNeumannNeighborhood<DIMENSION>::total>opencal::CALVonNeumannNeighborhood<DIMENSION>::indices;

} // namespace opencal



#endif // OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
