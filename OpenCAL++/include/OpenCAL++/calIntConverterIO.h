#ifndef calIntConverterIO_h
#define calIntConverterIO_h

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calConverterIO.h>

/*! \brief Derived class of CALConverterIO that implements virtual methods for I/O conversion of integer element.
*/

class CALIntConverterIO : public CALConverterIO
{
protected:
    /*! \brief Converts string to a floating point object.
    */
    virtual void* convertInput (std::string input);

    /*! \brief Converts floating point to string.
    */
    virtual std::string convertOutput (void* output);


};

void* CALIntConverterIO :: convertInput(std::string input)
{
    int* converted = new int (std::stoi(input));
    return (void*) (converted);
}


std::string CALIntConverterIO :: convertOutput(void* output)
{
   int* toConvert = (int*) output;
   std::string converted = std::to_string(*toConvert);

   return converted;

}


#endif
