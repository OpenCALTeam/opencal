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

%module(directors="1") calModelFunctor
%{
	#include "CalModelFunctor.h"
%}

%include "CalModelFunctor.h"

%feature("director") CalModelFunctor<CALModel3D,CALbyte>;
%template(StopConditionFunction3D) CalModelFunctor<CALModel3D,CALbyte>;

%feature("director") CalModelFunctor<CALModel2D,CALbyte>;
%template(StopConditionFunction2D) CalModelFunctor<CALModel2D,CALbyte>;

%feature("director") CalModelFunctor<CALModel2D,void>;
%template(CalModelFunctor2D) CalModelFunctor<CALModel2D,void>;

%feature("director") CalModelFunctor<CALModel3D,void>;
%template(CalModelFunctor3D) CalModelFunctor<CALModel3D,void>;
