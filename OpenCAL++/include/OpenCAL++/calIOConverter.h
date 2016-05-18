#ifndef OPENCAL_IOCONVERTER_H
#define OPENCAL_IOCONVERTER_H
#include <string>

namespace opencal {
// interface for all singleton converters
template <typename T>
class Converter {
 public:
  static auto& getInstance();

  virtual T deserialize(const std::string& input) = 0;

  virtual std::string serialize(const T& output) = 0;

  Converter(Converter const&) = delete;
  void operator=(Converter const&) = delete;
  Converter() = default;

  std::string operator()(const T& p) { return serialize(p); };

  T operator()(const std::string& s) { return deserialize(s); };
};

}


#endif //OPENCAL_IOCONVERTER_H
