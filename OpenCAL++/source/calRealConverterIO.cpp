<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calRealConverterIO.h>
=======
#include <OpenCAL++11/calRealConverterIO.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calRealConverterIO.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

void* CALRealConverterIO :: convertInput(std::string input)
{
    calCommon:: CALreal * converted = new calCommon:: CALreal (std::stof(input));
    return (void*) (converted);
}


std::string CALRealConverterIO :: convertOutput(void* output)
{
   calCommon:: CALreal* toConvert = (calCommon:: CALreal*) output;

   std::string converted = std::to_string(*toConvert);

   return converted;

}
