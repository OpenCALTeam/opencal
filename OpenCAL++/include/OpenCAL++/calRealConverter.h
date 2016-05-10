 
#ifndef OPENCAL_ALL_CALREALCONVERTERIO_H
#define OPENCAL_ALL_CALREALCONVERTERIO_H


#include <OpenCAL++/calCommon.h>

#include <string>
#include <sstream>
#pragma once

namespace opencal {


class CALRealConverter {
public:
    double convertInput(std::string input)
    {
        return (std::stof(input));
    }


    std::string convertOutput(double output)
    {
       std::string converted = std::to_string(output);
       return converted;

    }
};
}
#endif
