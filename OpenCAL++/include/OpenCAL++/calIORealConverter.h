#ifndef OPENCAL_ALL_CALREALCONVERTERIO_H
#define OPENCAL_ALL_CALREALCONVERTERIO_H

#include <OpenCAL++/calCommon.h>
#include<OpenCAL++/calIOConverter.h>
#include <string>

#pragma once

namespace opencal {

class CALRealConverter : public Converter<double> {
  using Converter::Converter;

 protected:
 public:
  
  double deserialize(const std::string& input) { return (std::stof(input)); }

  std::string serialize(const double& output) {
    std::string converted = std::to_string(output);
    return converted;
  }
};



}  // namespace opencal
#endif
