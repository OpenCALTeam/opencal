/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef calConverterI0_h
#define calConverterIO_h

#include <OpenCAL++/calCommon.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#pragma once

class CALConverterIO
{
public:

    /*! \brief loads a matrix from file.
    */
    template <class T>
    T* loadBuffer (int size, char* path);
    template <class T>
    T* loadBuffer (int* coordinates, size_t dimension,  char* path);

    /*! \brief saves a certain matrix to file.
    */
    template <class T>
    void calSaveBuffer(T* buffer, int size, int* coordinates, size_t dimension, char* path);

protected:

    /*! \brief Converts string to a certain object.
    */
    virtual void* convertInput (std::string input) = 0;

    /*! \brief Converts a given object to string.
    */
    virtual std::string convertOutput (void* output) = 0;

};

template <class T>
T* CALConverterIO :: loadBuffer (int size, char* path)
{
    T* buffer = new T [size];

    std::string line, token;
    std::ifstream in(path);
    int i = 0;

    while(getline(in, line))
    {
        std::stringstream s(line);
        while (s >> token)
        {
           T* converted = (T*) convertInput(token);
           buffer[i] =(*converted);

           delete converted;
           i ++;
        }
    }
    return buffer;
}

template <class T>
T* CALConverterIO :: loadBuffer (int* coordinates, size_t dimension,  char* path)
{
    T* buffer = new T [calCommon:: multiplier (coordinates, 0, dimension)];

    std::string line, token;
    std::ifstream in(path);
    int i = 0;

    while(getline(in, line))
    {
        std::stringstream s(line);
        while (s >> token)
        {
           i ++;
           T* converted = convertInput(token);
           buffer[i] = (T)(*converted);
           delete converted;
        }
    }
    return buffer;

}

template <class T>
void CALConverterIO :: calSaveBuffer(T* buffer, int size, int* coordinates, size_t dimension, char* path)
{
    std::ofstream out;
    out.open (path);
    for (int i = 0; i < size; ++i)
    {
        void* toConvert = &buffer[i];
        out<<convertOutput( toConvert);

        if ((i+1)% coordinates[1]==0)
            out<<'\n';
        else
        {
             out<<"  ";
        }
    }
    out.close();

}

#endif
