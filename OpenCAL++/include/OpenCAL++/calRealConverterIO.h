#ifndef calRealConverterIO_h
#define calRealConverterIO_h

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calConverterIO.h>


/*! \brief Derived class of CALConverterIO that implements virtual methods for I/O conversion of floating point element.
*/
class CALRealConverterIO : public CALConverterIO
{
protected:
    /*! \brief Converts string to a floating point object.
    */
    virtual void* convertInput (std::string input);

    /*! \brief Converts floating point to string.
    */
    virtual std::string convertOutput (void* output);


};


#endif
