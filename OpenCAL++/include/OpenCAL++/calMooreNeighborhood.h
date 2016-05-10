

#ifndef OPENCAL_CALMOORENEIGHBORHOOD_H
#define OPENCAL_CALMOORENEIGHBORHOOD_H


namespace opencal {
template<unsigned int DIMENSION>
class CALMooreNeighborhood {
protected:

  static std::array<uint, DIMENSION> indices;

  static void defineNeighborhood() {
    assert(DIMENSION > 1);
    const int alphabet[] = { 0, -1, 1 };

    constexpr const int total = calCommon::pow_ct(3, DIMENSION);

    for (int i = 0; i < total; i++)
    {
      int v = i;

      for (int pos = 0; pos < DIMENSION; pos++)
      {
        CALMooreNeighborhood<DIMENSION>::indices[pos] = alphabet[v % 3];
        v                                             = v / 3;
      }
    }
  }

public:

  CALMooreNeighborhood() {
    defineNeighborhood(); // create the neighbohood
  }

  static const auto&  getNeighborhoodIndices() {
    return indices;
  }
};

// static member declaration
template<uint DIMENSION>
std::array<uint, DIMENSION>opencal::CALMooreNeighborhood<DIMENSION>::indices;
} // namespace opencal

#endif // OPENCAL_CALMOORENEIGHBORHOOD_H
