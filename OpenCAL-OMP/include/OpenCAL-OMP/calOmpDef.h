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


	#define CAL_SET_CELL_LOCK(i, j, ca2D)					\
		omp_set_lock(&calGetMatrixElement((ca2D)->locks, (ca2D)->columns, (i), (j)));

	#define CAL_UNSET_CELL_LOCK(i, j, ca2D)					\
		omp_unset_lock(&calGetMatrixElement((ca2D)->locks, (ca2D)->columns, (i), (j)));

	#define CAL_LOCKS_DEFINE(name)			\
		omp_lock_t *name

	#define CAL_GET_THREAD_NUM()			\
		omp_get_thread_num()

	#define CAL_GET_NUM_THREADS()			\
		omp_get_num_threads()

	#define CAL_ALLOC_LOCKS(ca2D)						\
		(ca2D)->locks = (omp_lock_t *)malloc(sizeof(omp_lock_t) * (ca2D)->rows * (ca2D)->columns)

	#define CAL_INIT_LOCKS(ca2D, i)					\
		for (i = 0; i < ca2D->rows * ca2D->columns; i++)	\
			omp_init_lock(&ca2D->locks[i])

	#define CAL_DESTROY_LOCKS(ca2D, i)					\
		for (i = 0; (i) < (ca2D)->rows * (ca2D)->columns; (i)++)	\
			omp_destroy_lock(&((ca2D)->locks[i]));			\


	#define CAL_ALLOC_LOCKS_3D(ca3D)					\
		(ca3D)->locks = (omp_lock_t *)malloc(sizeof(omp_lock_t) * (ca3D)->rows * (ca3D)->columns * (ca3D)->slices)

	#define CAL_INIT_LOCKS_3D(ca3D, i)						\
		for (i = 0; i < (ca3D)->rows * (ca3D)->columns * (ca3D)->slices; i++) \
			omp_init_lock(&ca3D->locks[i])

	#define CAL_DESTROY_LOCKS_3D(ca3D, i)					\
		for (i = 0; (i) < (ca3D)->rows * (ca3D)->columns * (ca3D)->slices; (i)++) \
			omp_destroy_lock(&((ca3D)->locks[i]));			\

	#define CAL_SET_CELL_LOCK_3D(i, j, k, ca3D)				\
		omp_set_lock(&calGetBuffer3DElement((ca3D)->locks, (ca3D)->rows, (ca3D)->columns, (i), (j), (k)));

	#define CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D)				\
		omp_unset_lock(&calGetBuffer3DElement((ca3D)->locks, (ca3D)->rows, (ca3D)->columns, (i), (j), (k)));

	#define CAL_FREE_LOCKS(ca2D)			\
		free((ca2D)->locks)
		
		
	#define CAL_GET_NUM_PROCS() 	\
		omp_get_num_procs()
		
	#define CAL_SET_NUM_THREADS(n)  \
			 omp_set_num_threads((n))
		
	#else

	#define CAL_SET_CELL_LOCK(i, j, ca2D)
	#define CAL_UNSET_CELL_LOCK(i, j, ca2D)
	#define CAL_LOCKS_DEFINE(name)
	#define CAL_GET_THREAD_NUM() 0
	#define CAL_GET_NUM_THREADS() 1
	#define CAL_ALLOC_LOCKS(ca2D)
	#define CAL_INIT_LOCKS(ca2D, i)
	#define CAL_DESTROY_LOCKS(ca2D, i)
	#define CAL_FREE_LOCKS(ca2D)
	#define CAL_ALLOC_LOCKS_3D(ca3D)
	#define CAL_INIT_LOCKS_3D(ca3D, i)
	#define CAL_DESTROY_LOCKS_3D(ca3D, i)
	#define CAL_SET_CELL_LOCK_3D(i, j, k, ca3D)
	#define CAL_UNSET_CELL_LOCK_3D(i, j, k, ca3D)
	#define CAL_GET_NUM_PROCS() 1
	#define CAL_SET_NUM_THREADS(n)

	#endif //ifdef CAL_OMP

#endif //ifndef calOmpDef_h
