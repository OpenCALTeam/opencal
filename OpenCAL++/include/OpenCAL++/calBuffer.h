//
// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_CALBUFFER_H
#define OPENCAL_ALL_CALBUFFER_H


#include <OpenCAL++/calCommon.h>
#include <cassert>
#include <memory>
#include <iostream>
#include <OpenCAL++/calManagerIO.h>

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

    template<class CALCONVERTER, class STR_TYPE = std::string>
    CALBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, const STR_TYPE& path, CALCONVERTER& calConverterInputOutput){
        this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates , 0 , DIMENSION);
        this-> buffer = CALManagerIO<DIMENSION, COORDINATE_TYPE> :: template loadBuffer<PAYLOAD>(this->size, calConverterInputOutput, path);
    }

    ~CALBuffer (){
        delete [] buffer;
    }
    void setBuffer (PAYLOAD* _buffer, int _size){
        this->buffer = _buffer;
        this->size = _size;
    }
    void setBuffer (PAYLOAD value){

        for (calCommon::uint i = 0;  i < size; i++)
            buffer[i] = value;

    }
    PAYLOAD getElement (std::array <COORDINATE_TYPE, DIMENSION>& indexes, std::array <COORDINATE_TYPE, DIMENSION>& coordinates)
    {
        return buffer[calCommon :: cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes,coordinates)];
    }

    PAYLOAD getElement (int linearIndex)
    {
        return buffer[linearIndex];
    }

    int getSize ()
    {
        return size;
    }

    void setElement (std::array <COORDINATE_TYPE, DIMENSION>& indexes, std::array <COORDINATE_TYPE, DIMENSION>& coordinates, PAYLOAD value)
    {
        calCommon::uint linearIndex= calCommon :: cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes,coordinates);
        buffer[linearIndex] = value;
    }

    void setElement (int linearIndex, PAYLOAD value)
    {
        buffer[linearIndex] = value;
    }

    template<class CALCONVERTER, class STR_TYPE = std::string>
    void saveBuffer (std::array <COORDINATE_TYPE, DIMENSION>& coordinates, CALCONVERTER& calConverterInputOutput, const STR_TYPE& path)
    {
        opencal::CALManagerIO<DIMENSION, COORDINATE_TYPE>:: template saveBuffer<PAYLOAD,CALCONVERTER>(this->buffer, this->size, coordinates, calConverterInputOutput, path);
    }



    void copyActiveCellsBuffer (BUFFER_TYPE_PTR M_src, int* activeCells, int sizeof_active_cells)
    {
        calCommon::uint c, n;

        for(n=0; n<sizeof_active_cells; n++)
        {
            c= activeCells[n];
            if (this->buffer[c] != (*M_src)[c])
                this->buffer[c] = (*M_src)[c];
        }
    }

    void setActiveCellsBuffer (int* activeCells, int sizeof_active_cells, PAYLOAD value)
    {
        calCommon::uint n;
        for(n=0; n<sizeof_active_cells; n++)
        {
            this->buffer[activeCells[n]] = value;
        }

    }

    void stampa (std::array <COORDINATE_TYPE, DIMENSION>& coordinates)
    {
        for(int i= 0; i< size; i++)
        {
            std::cout<<this->buffer[i];
            if ((i+1) % coordinates[DIMENSION-1] == 0)
            {
                std::cout<<'\n';
            }
            else
            {
                std::cout<<"  ";
            }
        }
    }

    PAYLOAD &operator[](int i)
    {
        assert (i < this-> size);
        return buffer [i];
    }

    BUFFER_TYPE operator+(const BUFFER_TYPE_REF & b)
    {
        assert (this->size == b.size);
        PAYLOAD* buffer = new PAYLOAD [this->size];
        for (int i = 0; i < b.size; ++i) {
            buffer[i] = this->buffer[i] + b.buffer[i];
        }
        return new BUFFER_TYPE (buffer, this->size);
    }

    BUFFER_TYPE operator-(const BUFFER_TYPE_REF & b)
    {
        assert (this->size == b.size);
        PAYLOAD* buffer = new PAYLOAD [this->size];
        for (int i = 0; i < b.size; ++i) {
            buffer[i] = this->buffer[i] - b.buffer[i];
        }
        return new BUFFER_TYPE (buffer, this->size);
    }

    BUFFER_TYPE_REF operator+=(const BUFFER_TYPE_REF& b)
    {
        assert (this->size == b.size);
        for (int i = 0; i < b.size; ++i) {
            this->buffer[i] = this->buffer[i] + b.buffer[i];
        }
        return *this;
    }

    BUFFER_TYPE_REF operator-=(const BUFFER_TYPE_REF & b)
    {
        assert (this->size == b.size);
        for (int i = 0; i < b.size; ++i) {
            this->buffer[i] = this->buffer[i] - b.buffer[i];
        }
        return *this;
    }

    BUFFER_TYPE_REF operator=(const BUFFER_TYPE_REF & b)
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
    }



};

}// namespace opencal



#endif //OPENCAL_ALL_CALBUFFER_H
