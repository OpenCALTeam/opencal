//
// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_CALBUFFER_H
#define OPENCAL_ALL_CALBUFFER_H


#include <OpenCAL++/calCommon.h>
#include <cassert>
#include <memory>

namespace opencal {

template <class PAYLOAD , uint DIMENSION, typename COORDINATE_TYPE = uint>
class CALBuffer
{
    typedef CALBuffer<PAYLOAD , DIMENSION, COORDINATE_TYPE> BUFFER_TYPE;
    typedef CALBuffer<PAYLOAD , DIMENSION, COORDINATE_TYPE>& BUFFER_TYPE_REF;
    typedef CALBuffer<PAYLOAD , DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_PTR;
private:
    PAYLOAD* buffer;
    int size;

public:
    CALBuffer (): size(0) , buffer(nullptr){
            this->size = 0;
            this->buffer = NULL;
    }
    CALBuffer (PAYLOAD* _buffer, int _size): buffer(_buffer) , size(_size){}

    CALBuffer (int _size) : size(_size){
        this-> buffer = new PAYLOAD [size];
    }




    CALBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates){
        this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates , 0 , DIMENSION);
        buffer = new PAYLOAD [size];
    }

    template<class CALCONVERTER>
    CALBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates,  std::string& path, CALCONVERTER* calConverterInputOutput){
        this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates , 0 , DIMENSION);
       // this-> buffer = calConverterInputOutput -> loadBuffer<PAYLOAD>(this->size, path);
    }

    ~CALBuffer (){
        delete [] buffer;
    }
    void setBuffer (PAYLOAD* _buffer, int _size){
        this->buffer = _buffer;
        this->size = _size;
    }
    void setBuffer (PAYLOAD value){

        for (uint i = 0;  i < size; i++)
            buffer[i] = value;

    }
    PAYLOAD getElement (int* indexes, std::array <COORDINATE_TYPE, DIMENSION>& coordinates);
    PAYLOAD getElement (int linearIndex);
    int getSize ();
    void setElement (int* indexes, std::array <COORDINATE_TYPE, DIMENSION>& coordinates, int dimension, PAYLOAD value);
    void setElement (int linearIndex, PAYLOAD value);
    void setSize (int size);

    template<class CALCONVERTER>
    void saveBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, CALCONVERTER* calConverterInputOutput, std::string* path);

    void copyActiveCellsBuffer (CALBuffer<PAYLOAD , DIMENSION, COORDINATE_TYPE>* M_src, int* activeCells, int sizeof_active_cells);
    void setActiveCellsBuffer (int* activeCells, int sizeof_active_cells, PAYLOAD value);
    void stampa (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, size_t dimension);
    PAYLOAD &operator[](int i);
    BUFFER_TYPE operator+(const BUFFER_TYPE_REF & b);
    BUFFER_TYPE operator-(const BUFFER_TYPE_REF & b);
    BUFFER_TYPE_REF operator+=(const BUFFER_TYPE_REF& b);
    BUFFER_TYPE_REF operator-=(const BUFFER_TYPE_REF & b);
    BUFFER_TYPE_REF operator=(const BUFFER_TYPE_REF & b);



};

/*





template<class PAYLOAD>
CALBuffer<PAYLOAD> :: CALBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, size_t dimension,  char* path, CALConverterIO * calConverterInputOutput)
{
    this->size = calCommon :: multiplier(coordinates, 0, dimension);
    this-> buffer = calConverterInputOutput-> loadBuffer<PAYLOAD>(this->size, path);
}

template <class PAYLOAD>
CALBuffer<PAYLOAD> :: ~ CALBuffer ()
{
    delete [] buffer;
}

template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: setBuffer (PAYLOAD* buffer, int size)
{
    this->buffer = buffer;
    this->size = size;
}

template <class PAYLOAD>
void CALBuffer <PAYLOAD> ::  setBuffer (PAYLOAD value)
{
    int i;
    for (i = 0;  i< size; i++)
    {
        buffer[i] = value;
    }
}

template <class PAYLOAD>
PAYLOAD CALBuffer<PAYLOAD> :: getElement (int *indexes, int *coordinates, int dimension)
{

    return buffer[calCommon :: cellLinearIndex(indexes,coordinates,dimension)];

}

template <class PAYLOAD>
PAYLOAD CALBuffer<PAYLOAD> :: getElement (int linearIndex)
{

    return buffer[linearIndex];

}

template<class PAYLOAD>
int CALBuffer<PAYLOAD> :: getSize()
{
    return size;
}

template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: setElement (int* indexes, std::array <COORDINATE_TYPE, DIMENSION>& coordinates, int dimension, PAYLOAD value)
{
    int linearIndex = calCommon :: cellLinearIndex(indexes,coordinates,dimension);

    buffer[linearIndex] = value;
}

template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: setElement (int linearIndex, PAYLOAD value)
{
    buffer[linearIndex] = value;
}



template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: setSize (int size)
{
    this->size = size;
}

template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: stampa(std::array <COORDINATE_TYPE, DIMENSION>& coordinates, size_t dimension)
{
    for(int i= 0; i< size; i++)
    {
        std::cout<<this->buffer[i];
        if ((i+1) % coordinates[dimension-1] == 0)
        {
            std::cout<<'\n';
        }
        else
        {
            std::cout<<"  ";
        }
    }
}

template<class PAYLOAD>
void CALBuffer<PAYLOAD> :: copyActiveCellsBuffer (CALBuffer <PAYLOAD>* M_src, int* activeCells, int sizeof_active_cells)
{
    int c, n;

    for(n=0; n<sizeof_active_cells; n++)
    {
        c= activeCells[n];
        if (this->buffer[c] != (*M_src)[c])
            this->buffer[c] = (*M_src)[c];
    }
}

template <class PAYLOAD>
void CALBuffer<PAYLOAD> :: setActiveCellsBuffer (int* cells, int sizeof_active_cells, PAYLOAD value)
{
    int n;
    for(n=0; n<sizeof_active_cells; n++)
    {
        this->buffer[cells[n]] = value;
    }
}
template <class PAYLOAD>
void CALBuffer<PAYLOAD>:: saveBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path)
{
    calConverterInputOutput->calSaveBuffer(this->buffer, this->size, coordinates, dimension, path);

}

template <class PAYLOAD>
PAYLOAD& CALBuffer<PAYLOAD> :: operator [] (int i)
{
    assert (i < this-> size);
    return buffer [i];
}

template <class PAYLOAD>
CALBuffer<PAYLOAD> CALBuffer<PAYLOAD> :: operator+(const CALBuffer<PAYLOAD> & b)
{
    assert (this->size == b.size);
    PAYLOAD* buffer = new PAYLOAD [this->size];
    for (int i = 0; i < b.size; ++i) {
        buffer[i] = this->buffer[i] + b.buffer[i];
    }
    return new CALBuffer<PAYLOAD> (buffer, this->size);

}

template<class PAYLOAD>
CALBuffer<PAYLOAD> CALBuffer<PAYLOAD> :: operator-(const CALBuffer<PAYLOAD>& b)
{
    assert (this->size == b.size);
    PAYLOAD* buffer = new PAYLOAD [this->size];
    for (int i = 0; i < b.size; ++i) {
        buffer[i] = this->buffer[i] - b.buffer[i];
    }
    return new CALBuffer<PAYLOAD> (buffer, this->size);

}

template <class PAYLOAD>
CALBuffer<PAYLOAD>& CALBuffer<PAYLOAD> :: operator+=(const CALBuffer<PAYLOAD> & b)
{
    assert (this->size == b.size);
    for (int i = 0; i < b.size; ++i) {
        this->buffer[i] = this->buffer[i] + b.buffer[i];
    }
    return *this;

}

template<class PAYLOAD>
CALBuffer<PAYLOAD>& CALBuffer<PAYLOAD> :: operator-=(const CALBuffer<PAYLOAD>& b)
{
    assert (this->size == b.size);
    for (int i = 0; i < b.size; ++i) {
        this->buffer[i] = this->buffer[i] - b.buffer[i];
    }
    return *this;

}

template<class PAYLOAD>
CALBuffer<PAYLOAD>& CALBuffer<PAYLOAD> :: operator=(const CALBuffer<PAYLOAD> & b)
{
    if (this != &b)
    {
        if (this->size == b.size)
        {
            for (int i = 0; i < b.size; ++i)
            {
                this->buffer[i] = b.buffer[i];
            }

        }else
        {
            PAYLOAD* bufferTmp = new PAYLOAD[b.size];

            for (int i = 0; i < b.size; ++i)
            {
                this->buffer[i] = b.buffer[i];
            }
            if (this->buffer)
                delete[] this->buffer;
            this->buffer = bufferTmp;
            this->size = b.size;
        }
    }
    return *this;


}*/

}// namespace opencal

#endif //OPENCAL_ALL_CALBUFFER_H
