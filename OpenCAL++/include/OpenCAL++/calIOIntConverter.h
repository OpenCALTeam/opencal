#ifndef OPENCAL_ALL_CALINTCONVERTERIO_H
#define OPENCAL_ALL_CALINTCONVERTERIO_H


#include<OpenCAL++/calIOConverter.h>
#include <iostream>
#include <fstream>
#include <string>

#pragma once

namespace opencal {

    class CALIntConverter : public Converter<int> {
     public:
        int deserialize(const std::string& input) { return (std::stof(input)); }

        std::string serialize(const int& output) {
        std::string converted = std::to_string(output);
        return converted;
      }
    };

}
#endif
