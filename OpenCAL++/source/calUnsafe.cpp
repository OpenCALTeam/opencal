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

#include <OpenCAL++/calUnsafe.h>

#include <stdlib.h>


CALUnsafe ::CALUnsafe (CALModel* calModel)
{
    this->calModel = calModel;
}



void CALUnsafe:: calAddActiveCellX (int* indexes, int n)
{


    int linearIndex = calCommon::cellLinearIndex(indexes, this->calModel->getCoordinates(),this->calModel->getDimension());
    int neighboorIndex = CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);


    if (!this->calModel->getActiveCells()->getFlag(neighboorIndex))
    {
        this->calModel->getActiveCells()->setFlag(neighboorIndex, true);

    }

}


void CALUnsafe:: calAddActiveCellX (int linearIndex, int n)
{

    int neighboorIndex =  CALNeighborPool::getInstance()->getNeighborN(linearIndex,n);


    if (!this->calModel->getActiveCells()->getFlag(neighboorIndex))
    {
        this->calModel->getActiveCells()->setFlag(neighboorIndex, true);

    }

}
