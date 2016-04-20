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

%module opencal
%{
#include<cal2DReduction.h>
#include<cal2DUnsafe.h>
#include<calCommon.h>
#include<cal2DRun.h>
#include<cal3DReduction.h>
#include<cal3DUnsafe.h>
#include<cal3DRun.h>

%}
%include carrays.i
%include "calCommon.i"

%include "cal2DBuffer.i"
%include "cal2DBufferIO.i"
%include "cal2DReduction.i"
%include "cal2DUnsafe.i"
%include "cal2DIO.i"
%include "cal2D.i"
%include "cal2DRun.i"
%include "ElementaryProcessFunctor2D.i"

%include "cal3DBuffer.i"
%include "cal3DBufferIO.i"
%include "cal3DUnsafe.i"
//%include "cal3DReduction.i"
%include "cal3DIO.i"
%include "cal3D.i"
%include "cal3DRun.i"
%include "ElementaryProcessFunctor3D.i"
%include "CalModelFunctor.i"
