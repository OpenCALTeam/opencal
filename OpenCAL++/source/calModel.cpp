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

#include <OpenCAL++/calModel.h>


/******************************************************************************
                            PUBLIC FUNCIONS

*******************************************************************************/


//----------------------------------------------
CALModel::CALModel(int* coordinates,
                   size_t dimension,
                   CALNeighborhood* calNeighborhood,
                   enum calCommon:: CALSpaceBoundaryCondition CAL_TOROIDALITY,
                   enum calCommon:: CALOptimization CAL_OPTIMIZATION
                   )
{

    this->coordinates = coordinates;
    this->dimension = dimension;
    this->CAL_TOROIDALITY = CAL_TOROIDALITY;

    this->size = calCommon:: multiplier(coordinates,0, dimension);

    CALIndexesPool::init(calCommon:: multiplier(this->coordinates, 0, this->dimension), coordinates, dimension);

    CALNeighborPool:: init(this->size, this->coordinates, this->dimension, CAL_TOROIDALITY);
    this->sizeof_X = 0;



    this->OPTIMIZATION = CAL_OPTIMIZATION;
    if (this->OPTIMIZATION == calCommon::CAL_OPT_ACTIVE_CELLS) {
        CALBuffer <bool>* flags = new CALBuffer<bool> (this->coordinates, this->dimension);
        flags->setBuffer(false);
        this->activeCells = new CALActiveCells (flags, 0);
    }

    else
        this->activeCells = NULL;

    this->pQ_arrays = NULL;
    this->sizeof_pQ_arrays = 0;

    this->X_id = calNeighborhood;
    if (X_id != NULL)
        this->X_id->defineNeighborhood(this);

    this->elementary_processes = NULL;
    this->num_of_elementary_processes = 0;
}
//----------------------------------------------

CALModel :: ~ CALModel ()
{
    for (int i = 0; i < this->sizeof_pQ_arrays; ++i) {
        delete pQ_arrays[i];
    }
    delete [] pQ_arrays;

    CALIndexesPool::destroy();
    CALNeighborPool::destroy();
    delete activeCells;
    delete coordinates;
    delete X_id;

    delete [] this->elementary_processes;

}

void CALModel::addActiveCell(int * indexes)
{
    this->activeCells->setElementFlags(indexes, this->coordinates, this->dimension, CAL_TRUE);
}

void CALModel::addActiveCell(int linearIndex)
{
    this->activeCells->setFlag(linearIndex, CAL_TRUE);
}

void CALModel::removeActiveCell(int * indexes)
{
    this->activeCells->setElementFlags(indexes, this->coordinates, this->dimension, CAL_FALSE);
}

void CALModel::removeActiveCell(int linearIndex)
{
    this->activeCells->setFlag(linearIndex, CAL_FALSE);
}

void CALModel::updateActiveCells()
{
    activeCells->update();
}



void CALModel::addNeighbor(int* indexes) {
     CALNeighborPool::getInstance()->addNeighbor(indexes);

    this->sizeof_X++;
}

void CALModel::addNeighbors(int** indexes, size_t size) {

    int n = 0;
    for (n = 0; n < size; n++){
        CALNeighborPool::getInstance()->addNeighbor(indexes[n]);
        this->sizeof_X ++;
    }
}


void CALModel::addElementaryProcess(CALCallbackFunc elementary_process)
{
    CALCallbackFunc* callbacks_temp = this->elementary_processes;
    CALCallbackFunc* callbacks_new = new CALCallbackFunc [this->num_of_elementary_processes + 1];

    int n;

    for (n = 0; n < this->num_of_elementary_processes; n++)
        callbacks_new[n] = this->elementary_processes[n];
    callbacks_new[this->num_of_elementary_processes] = elementary_process;

    this->elementary_processes = callbacks_new;
    delete [] callbacks_temp;

    this->num_of_elementary_processes++;

}


void CALModel::applyElementaryProcess(CALCallbackFunc elementary_process //!< Pointer to a transition function's elementary process
                                         )
{
    int i, n;

    if (this->activeCells) //Computationally active cells optimization.
    {
        int sizeCurrent = this->activeCells->getSizeCurrent();
        for (n=0; n<sizeCurrent; n++)
            elementary_process->run(this, calCommon:: cellMultidimensionalIndexes(this->activeCells->getCells()[n]));
    }
    else //Standart cicle of the transition function
    {

        for (i=0; i<this->size; i++)
        {
            int * indexes = calCommon:: cellMultidimensionalIndexes(i);
            elementary_process->run(this, indexes);
        }

    }
}


void CALModel::globalTransitionFunction()
{
    //The global transition function.
    //It applies transition function elementary processes sequentially.
    //Note that a substates' update is performed after each elementary process.

    int b;

    for (b=0; b<this->num_of_elementary_processes; b++)
    {
        //applying the b-th elementary process
        this->applyElementaryProcess(this->elementary_processes[b]);
        //updating substates
        this-> update();
    }
}



void CALModel::update()
{
    //updating active cells
    if (this->OPTIMIZATION == calCommon :: CAL_OPT_ACTIVE_CELLS)
        this->updateActiveCells();


    for (int i = 0; i < this->sizeof_pQ_arrays; ++i)
    {
        pQ_arrays[i]->update(this->activeCells);
    }


}

size_t CALModel:: getDimension ()
{
    return this->dimension;
}

int CALModel::getSize ()
{
    return this->size;
}

int* CALModel::getCoordinates ()
{
    return this->coordinates;
}


int CALModel:: getNeighborhoodSize ()
{
    return this->sizeof_X;
}


CALActiveCells* CALModel :: getActiveCells ()
{
    return activeCells;
}


void CALModel::setNeighborhoodSize (int sizeof_X)
{
    this->sizeof_X = sizeof_X;
}
