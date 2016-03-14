// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calUnsafe.h>
=======
#include <OpenCAL++11/calUnsafe.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calUnsafe.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
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
