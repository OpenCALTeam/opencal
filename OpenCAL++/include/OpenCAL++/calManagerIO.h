//
// Created by knotman on 13/04/16.
//

#ifndef OPENCAL_ALL_CALCONVERTERIO_H
#define OPENCAL_ALL_CALCONVERTERIO_H


#include <OpenCAL++/calCommon.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#pragma once

namespace opencal {

    template<uint DIMENSION, typename COORDINATE_TYPE = uint>
    class CALManagerIO {
       // typedef std::function< T*( int, STR_TYPE ) > Function;

    public:

        /*! \brief loads a matrix from file.
        */
        template<class T, class CALCONVERTER, class STR_TYPE = std::string>
        static T *loadBuffer(int size, CALCONVERTER& converter, const STR_TYPE& path){
            T *buffer = new T[size];

            std::string line, token;
            std::ifstream in(path);
            int i = 0;

            while (getline(in, line)) {
                std::stringstream s(line);
                while (s >> token) {
                    T converted = converter(token);
                    buffer[i] = converted;

                    i++;
                }
            }
            return buffer;
        }



        template<typename T,class STR_TYPE > inline
        static T *loadBuffer(int size, std::function<T*(int, const std::string&)>& loadCallback, const STR_TYPE& path){
            T *buffer = loadCallback(size,path);

            return buffer;
        }
        template<class T, class CALCONVERTER, class STR_TYPE = std::string>
        inline static T *loadBuffer(std::array <COORDINATE_TYPE, DIMENSION>& coordinates, CALCONVERTER converter, const STR_TYPE& path){
            return loadBuffer(opencal::calCommon::multiplier(coordinates, 0, DIMENSION),converter,path);
        }

        /*! \brief saves a certain matrix to file.
        */
        template<class T, class CALCONVERTER, class STR_TYPE = std::string>
        static void  saveBuffer(T *buffer, int size, std::array <COORDINATE_TYPE, DIMENSION>& coordinates,CALCONVERTER& converter, const STR_TYPE& path) {
            std::ofstream out;
            out.open(path);
            for (int i = 0; i < size; ++i) {
                out << converter(buffer[i]);

                if ((i + 1) % coordinates[1] == 0)
                    out << '\n';
                else
                    out << "  ";

                int dim = size;
                for (uint j = DIMENSION-1; j> 1; j--)
                {
                    dim = dim/coordinates[j];
                    if ((i + 1) % dim == 0)
                        out << '\n';
                }
            }
            out.close();
        }



        template<typename T,class STR_TYPE > inline
        static void saveBuffer(T *buffer, int size, std::array <COORDINATE_TYPE, DIMENSION>& coordinates,
                                std::function<void(const T*, const std::string&)>& saveCallback, const STR_TYPE& path){

            saveCallback(buffer,path);


        }




    };



}//namespace opencal

#endif //OPENCAL_ALL_CALCONVERTERIO_H
