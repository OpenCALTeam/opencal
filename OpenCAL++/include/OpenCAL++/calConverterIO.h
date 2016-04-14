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
    class CALConverterIO {
    public:

        /*! \brief loads a matrix from file.
        */
        template<class T>
        T *loadBuffer(int size, char *path){
            T *buffer = new T[size];

            std::string line, token;
            std::ifstream in(path);
            int i = 0;

            while (getline(in, line)) {
                std::stringstream s(line);
                while (s >> token) {
                    T *converted = (T *) convertInput(token);
                    buffer[i] = (*converted);

                    delete converted;
                    i++;
                }
            }
            return buffer;
        }

        template<class T>
        T *loadBuffer(std::array <COORDINATE_TYPE, DIMENSION>& coordinates, size_t dimension, char *path){
            T *buffer = new T[opencal::calCommon::multiplier(coordinates, 0, DIMENSION)];

            std::string line, token;
            std::ifstream in(path);
            int i = 0;

            while (getline(in, line)) {
                std::stringstream s(line);
                while (s >> token) {
                    i++;
                    T *converted = convertInput(token);
                    buffer[i] = (T) (*converted);
                    delete converted;
                }
            }
            return buffer;
        }

        /*! \brief saves a certain matrix to file.
        */
        template<class T>
        void calSaveBuffer(T *buffer, int size, std::array <COORDINATE_TYPE, DIMENSION>& coordinates, char *path) {
            std::ofstream out;
            out.open(path);
            for (int i = 0; i < size; ++i) {
                void *toConvert = &buffer[i];
                out << convertOutput(toConvert);

                if ((i + 1) % coordinates[1] == 0)
                    out << '\n';
                else {
                    out << "  ";
                }
            }
            out.close();
        }

    protected:

        /*! \brief Converts string to a certain object.
        */
        virtual void *convertInput(std::string input) = 0;

        /*! \brief Converts a given object to string.
        */
        virtual std::string convertOutput(void *output) = 0;

    };



}//namespace opencal

#endif //OPENCAL_ALL_CALCONVERTERIO_H
