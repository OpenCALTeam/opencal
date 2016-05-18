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
   static const std::array<element, total>&  defineNeighborhood(){


       indices[0] = {0};//central cell
       //total number of insertions is 2*Dimension+1
       for (int i = 1; i <= DIMENSION; ++i)
       {
           indices[i] = {0};
           indices[i][i-1] = -1;
       }
       int c = DIMENSION -1;
       for (int i = DIMENSION+1; i <= 2*DIMENSION; ++i,--c)
       {
           indices[i] = {0};
           indices[i][c] = 1;
       }

       return indices;
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
                                                   { {-1, 0} },
                                                   { {0,-1} },
                                                   { {0, 1} },
                                                   { {1, 0} }
                                               } };
}*/
/*


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
std::array<typename CALVonNeumannNeighborhood<DIMENSION>::element, CALVonNeumannNeighborhood<DIMENSION>::total>
 opencal::CALVonNeumannNeighborhood<DIMENSION>::indices = opencal::CALVonNeumannNeighborhood<DIMENSION>::defineNeighborhood();

} // namespace opencal



#endif // OPENCAL_ALL_CALVONNEUMANNNEIGHBORHOOD_H
