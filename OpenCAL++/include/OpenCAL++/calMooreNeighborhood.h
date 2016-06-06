

#ifndef OPENCAL_CALMOORENEIGHBORHOOD_H
#define OPENCAL_CALMOORENEIGHBORHOOD_H


namespace opencal {
template<unsigned int DIMENSION,unsigned int RADIUS = 1>
class CALMooreNeighborhood {
protected:

  static constexpr const int ALPHABETHSIZE = 2*RADIUS+1;
  static constexpr const int total = calCommon::pow_ct(ALPHABETHSIZE, DIMENSION);
public:
    typedef int COORDINATE_TYPE;
protected:
  typedef std::array<COORDINATE_TYPE,DIMENSION> element;
   static std::array<element, total> indices;

public:
  static constexpr const unsigned int _DIMENSION = DIMENSION;

static const std::array<element, total>& defineNeighborhood() {
    assert(DIMENSION > 1);

    const  std::array<COORDINATE_TYPE ,ALPHABETHSIZE> alphabet = generateAlphabeth();

    for (int i = 0; i < total; i++)
     for (int pos = 0 , v=i ; pos < DIMENSION; pos++, v/=ALPHABETHSIZE)
        CALMooreNeighborhood<DIMENSION,RADIUS>::indices[i][pos] = alphabet[v % ALPHABETHSIZE];

     return indices;
  }

  static const auto&  getNeighborhoodIndices() {
       return indices;
  }

protected:
 static const std::array<COORDINATE_TYPE ,ALPHABETHSIZE> generateAlphabeth(){
    std::array<COORDINATE_TYPE , ALPHABETHSIZE> alphabet;
    alphabet[0] = 0;

    for(int i = 1 , pos = 1 ; i <= RADIUS ; i++ ){
        alphabet[pos++] = i;
        alphabet[pos++] = -i;

    }
    return alphabet;
  }
};

// static member declaration
template<unsigned int DIMENSION, unsigned int RADIUS>
  std::array<typename CALMooreNeighborhood<DIMENSION,RADIUS>::element, CALMooreNeighborhood<DIMENSION,RADIUS>::total>
            opencal::CALMooreNeighborhood<DIMENSION,RADIUS>::indices = opencal::CALMooreNeighborhood<DIMENSION,RADIUS>::defineNeighborhood();
} // namespace opencal

#endif // OPENCAL_CALMOORENEIGHBORHOOD_H
