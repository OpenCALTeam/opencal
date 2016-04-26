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

#ifndef calElementaryProcess_h_
#define calElementaryProcess_h_

class CALModel;

/*! \brief Class that defines transition function's elementary processes.
*/
class CALElementaryProcessFunctor {
public:

    CALElementaryProcessFunctor();

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
    */
    virtual void run(CALModel* calModel, int* indexes) =0;
    virtual void operator()(CALModel* calModel, int* indexes);
    virtual ~CALElementaryProcessFunctor();
};


#endif /* EMENTARYPROCESS_H_ */
