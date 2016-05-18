

#ifndef OPENCAL_CALMOORENEIGHBORHOOD_H
#define OPENCAL_CALMOORENEIGHBORHOOD_H


namespace opencal {
template<unsigned int DIMENSION>
class CALMooreNeighborhood {
protected:

static constexpr const int total = calCommon::pow_ct(3, DIMENSION);
 typedef std::array<int,DIMENSION> element;

 static std::array<element, total> indices;

public:
  static const std::array<element, total>& defineNeighborhood() {
    assert(DIMENSION > 1);
    const int alphabet[] = { 0, -1, 1 };


    for (int i = 0; i < total; i++)
    {
      int v = i;

      for (int pos = 0; pos < DIMENSION; pos++)
      {
        CALMooreNeighborhood<DIMENSION>::indices[i][pos] = alphabet[v % 3];
        v = v / 3;

      }
     }
     return indices;
  }

  static const auto&  getNeighborhoodIndices() {
    return indices;
  }
};

// static member declaration
template<uint DIMENSION>
std::array<typename CALMooreNeighborhood<DIMENSION>::element, CALMooreNeighborhood<DIMENSION>::total>
opencal::CALMooreNeighborhood<DIMENSION>::indices = opencal::CALMooreNeighborhood<DIMENSION>::defineNeighborhood();
} // namespace opencal

#endif // OPENCAL_CALMOORENEIGHBORHOOD_H
