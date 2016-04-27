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
    public:

        /*! \brief loads a matrix from file.
        */
        template<class T, class CALCONVERTER>
        static T *loadBuffer(int size, CALCONVERTER* converter, char *path){
            T *buffer = new T[size];

            std::string line, token;
            std::ifstream in(path);
            int i = 0;

            while (getline(in, line)) {
                std::stringstream s(line);
                while (s >> token) {
                    T converted = converter->convertInput(token);
                    buffer[i] = converted;

                    i++;
                }
            }
            return buffer;
        }

        template<class T, class CALCONVERTER>
        static T *loadBuffer(std::array <COORDINATE_TYPE, DIMENSION>& coordinates, CALCONVERTER* converter, char *path){
            T *buffer = new T[opencal::calCommon::multiplier(coordinates, 0, DIMENSION)];

            std::string line, token;
            std::ifstream in(path);
            int i = 0;

            while (getline(in, line)) {
                std::stringstream s(line);
                while (s >> token) {
                    i++;
                    T converted = converter->convertInput(token);
                    buffer[i] = converted;

                }
            }
            return buffer;
        }

        /*! \brief saves a certain matrix to file.
        */
        template<class T, class CALCONVERTER>
        static void  saveBuffer(T *buffer, int size, std::array <COORDINATE_TYPE, DIMENSION>& coordinates,CALCONVERTER* converter, char *path) {
            std::ofstream out;
            out.open(path);
            for (int i = 0; i < size; ++i) {
                out << converter->convertOutput(buffer[i]);

                if ((i + 1) % coordinates[1] == 0)
                    out << '\n';
                else {
                    out << "  ";
                }
            }
            out.close();
        }


    };



}//namespace opencal

#endif //OPENCAL_ALL_CALCONVERTERIO_H
