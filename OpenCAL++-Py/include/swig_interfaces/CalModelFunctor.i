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
