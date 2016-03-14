#ifndef calBuffer_h
#define calBuffer_h

#include <OpenCAL++11/calCommon.h>
#include <OpenCAL++11/calConverterIO.h>
#include <cassert>

template <class T>
class CALBuffer
{
private:
    T* buffer;
    int size;

public:
    CALBuffer (T* buffer, int size);
    CALBuffer (int size);
    CALBuffer (int* coordinates, size_t dimension);
    CALBuffer (int* coordinates, size_t dimension,  char* path, CALConverterIO * calConverterInputOutput);
    CALBuffer ();
    ~CALBuffer ();
    void setBuffer (T* buffer, int size);
    void setBuffer (T value);
    T getElement (int* indexes, int* coordinates, int dimension);
    T getElement (int linearIndex);
    int getSize ();
    void setElement (int* indexes, int* coordinates, int dimension, T value);
    void setElement (int linearIndex, T value);
    void setSize (int size);
    void saveBuffer (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path);
    void copyActiveCellsBuffer (CALBuffer <T>* M_src, int* activeCells, int sizeof_active_cells);
    void setActiveCellsBuffer (int* activeCells, int sizeof_active_cells, T value);
    void stampa (int* coordinates, size_t dimension);
    T &operator[](int i);
    CALBuffer<T> operator+(const CALBuffer<T> & b);
    CALBuffer<T> operator-(const CALBuffer<T> & b);
    CALBuffer<T>& operator+=(const CALBuffer<T> & b);
    CALBuffer<T>& operator-=(const CALBuffer<T> & b);
    CALBuffer<T>& operator=(const CALBuffer<T> & b);



};

template <class T>
CALBuffer<T> :: CALBuffer (T *buffer, int size)
{
    this->buffer = buffer;
    this->size = size;
}

template <class T>
CALBuffer<T> :: CALBuffer (int size)
{
    this->size = size;
    this-> buffer = new T [size];
}

template <class T>
CALBuffer<T> :: CALBuffer ()
{
    this->size = 0;
    this->buffer = NULL;
}

template<class T>
CALBuffer<T> :: CALBuffer (int* coordinates, size_t dimension)
{
    this->size = calCommon :: multiplier (coordinates, 0, dimension);
    buffer = new T [size];
}

template<class T>
CALBuffer<T> :: CALBuffer (int* coordinates, size_t dimension,  char* path, CALConverterIO * calConverterInputOutput)
{
    this->size = calCommon :: multiplier(coordinates, 0, dimension);
    this-> buffer = calConverterInputOutput-> loadBuffer<T>(this->size, path);
}

template <class T>
CALBuffer<T> :: ~ CALBuffer ()
{
    delete [] buffer;
}

template <class T>
void CALBuffer<T> :: setBuffer (T* buffer, int size)
{
    this->buffer = buffer;
    this->size = size;
}

template <class T>
void CALBuffer <T> ::  setBuffer (T value)
{
    int i;
    for (i = 0;  i< size; i++)
    {
        buffer[i] = value;
    }
}

template <class T>
T CALBuffer<T> :: getElement (int *indexes, int *coordinates, int dimension)
{

    return buffer[calCommon :: cellLinearIndex(indexes,coordinates,dimension)];

}

template <class T>
T CALBuffer<T> :: getElement (int linearIndex)
{

    return buffer[linearIndex];

}

template<class T>
int CALBuffer<T> :: getSize()
{
    return size;
}

template <class T>
void CALBuffer<T> :: setElement (int* indexes, int* coordinates, int dimension, T value)
{
    int linearIndex = calCommon :: cellLinearIndex(indexes,coordinates,dimension);

    buffer[linearIndex] = value;
}

template <class T>
void CALBuffer<T> :: setElement (int linearIndex, T value)
{
    buffer[linearIndex] = value;
}



template <class T>
void CALBuffer<T> :: setSize (int size)
{
    this->size = size;
}

template <class T>
void CALBuffer<T> :: stampa(int* coordinates, size_t dimension)
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

template<class T>
void CALBuffer<T> :: copyActiveCellsBuffer (CALBuffer <T>* M_src, int* activeCells, int sizeof_active_cells)
{
    int c, n;

    for(n=0; n<sizeof_active_cells; n++)
    {
        c= activeCells[n];
        if (this->buffer[c] != (*M_src)[c])
            this->buffer[c] = (*M_src)[c];
    }
}

template <class T>
void CALBuffer<T> :: setActiveCellsBuffer (int* cells, int sizeof_active_cells, T value)
{
    int n;
    for(n=0; n<sizeof_active_cells; n++)
    {
        this->buffer[cells[n]] = value;
    }
}
template <class T>
void CALBuffer<T>:: saveBuffer (int* coordinates, size_t dimension, CALConverterIO* calConverterInputOutput, char* path)
{
    calConverterInputOutput->calSaveBuffer(this->buffer, this->size, coordinates, dimension, path);

}

template <class T>
T& CALBuffer<T> :: operator [] (int i)
{
    assert (i < this-> size);
    return buffer [i];
}

template <class T>
CALBuffer<T> CALBuffer<T> :: operator+(const CALBuffer<T> & b)
{
    assert (this->size == b.size);
    T* buffer = new T [this->size];
    for (int i = 0; i < b.size; ++i) {
        buffer[i] = this->buffer[i] + b.buffer[i];
    }
    return new CALBuffer<T> (buffer, this->size);

}

template<class T>
CALBuffer<T> CALBuffer<T> :: operator-(const CALBuffer<T>& b)
{
    assert (this->size == b.size);
    T* buffer = new T [this->size];
    for (int i = 0; i < b.size; ++i) {
        buffer[i] = this->buffer[i] - b.buffer[i];
    }
    return new CALBuffer<T> (buffer, this->size);

}

template <class T>
CALBuffer<T>& CALBuffer<T> :: operator+=(const CALBuffer<T> & b)
{
    assert (this->size == b.size);
    for (int i = 0; i < b.size; ++i) {
        this->buffer[i] = this->buffer[i] + b.buffer[i];
    }
    return *this;

}

template<class T>
CALBuffer<T>& CALBuffer<T> :: operator-=(const CALBuffer<T>& b)
{
    assert (this->size == b.size);
    for (int i = 0; i < b.size; ++i) {
        this->buffer[i] = this->buffer[i] - b.buffer[i];
    }
    return *this;

}

template<class T>
CALBuffer<T>& CALBuffer<T> :: operator=(const CALBuffer<T> & b)
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
            T* bufferTmp = new T[b.size];

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

#endif

