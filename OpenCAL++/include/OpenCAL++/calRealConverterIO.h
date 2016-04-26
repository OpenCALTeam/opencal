/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
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
