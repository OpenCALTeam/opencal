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

#ifndef calOmpDef_h
#define calOmpDef_h

    enum CALUnsafeState {
        CAL_UNSAFE_ACTIVE,
        CAL_UNSAFE_INACTIVE
    };

    #ifdef HAVE_CONFIG_H
    #include<config.h>
    #endif

    #ifdef _OPENMP

    #include <omp.h>


    #define CAL_SET_CELL_LOCK(index, lock_matrix)					\
        omp_set_lock(lock_matrix[index]);

    #define CAL_UNSET_CELL_LOCK(index, lock_matrix)					\
        omp_unset_lock(lock_matrix[index]);

    #define CAL_LOCKS_DEFINE(name)			\
        omp_lock_t **name

    #define CAL_GET_THREAD_NUM()			\
        omp_get_thread_num()

    #define CAL_GET_NUM_THREADS()			\
        omp_get_num_threads()

    #define CAL_ALLOC_LOCKS(calModel, cellular_space_dimensions)						\
        (calModel)->locks = (omp_lock_t **)malloc(sizeof(omp_lock_t*) * cellular_space_dimension)

    #define CAL_INIT_LOCKS(calModel, i)					\
        for (i = 0; i < (calModel)->cellularSpaceDimension; i++)	\
            omp_init_lock(calModel->locks[i])

    #define CAL_DESTROY_LOCKS(calModel, i)					\
    for (i = 0; i < (calModel)->cellularSpaceDimension; i++)	\
        omp_destroy_lock(calModel->locks[i])		\

    #define CAL_FREE_LOCKS(calModel, i)			\
        for(i = 0; i < (calModel)->cellularSpaceDimension; i++)
            free((calModel)->locks[i])


    #define CAL_GET_NUM_PROCS() 	\
        omp_get_num_procs()

    #define CAL_SET_NUM_THREADS(n)  \
             omp_set_num_threads((n))

    #else

    #define CAL_SET_CELL_LOCK(index, lock_matrix)
    #define CAL_UNSET_CELL_LOCK(index, lock_matrix)
    #define CAL_LOCKS_DEFINE(name)
    #define CAL_GET_THREAD_NUM() 0
    #define CAL_GET_NUM_THREADS() 1
    #define CAL_ALLOC_LOCKS(calModel, cellular_space_dimensions)
    #define CAL_INIT_LOCKS(calModel, i)
    #define CAL_DESTROY_LOCKS(calModel, i)
    #define CAL_FREE_LOCKS(calModel, i)
    #define CAL_GET_NUM_PROCS() 1
    #define CAL_SET_NUM_THREADS(n)

    #endif //ifdef CAL_OMP

#endif //ifndef calOmpDef_h
