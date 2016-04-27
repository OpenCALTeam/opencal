#ifndef OPENCAL_ALL_CALINTCONVERTERIO_H
#define OPENCAL_ALL_CALINTCONVERTERIO_H


#include <OpenCAL++/calCommon.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#pragma once

namespace opencal {


class CALIntConverter {
public:
    int convertInput(std::string input)
    {
        return (std::stoi(input));
    }


    std::string convertOutput(int output)
    {
       std::string converted = std::to_string(output);
       return converted;

    }
};
}
#endif
