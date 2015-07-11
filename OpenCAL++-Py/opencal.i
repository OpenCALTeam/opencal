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
%include "cal3DReduction.i"
%include "cal3DIO.i"
%include "cal3D.i"
%include "cal3DRun.i"
%include "ElementaryProcessFunctor3D.i"
