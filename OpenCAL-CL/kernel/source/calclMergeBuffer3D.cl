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


#include <OpenCAL-CL/calcl3D.h>

__kernel void calclsetDiffFlags3D(
  CALint rows,
  CALint columns,
  CALint slices,
  CALint borderSize,
  CALint chunkSize,
   __global CALbyte* diff)
{

 /*int threadID = get_global_id (0);
  const int borderSizeReal= borderSize*columns;
  const int realSize = columns*rows;

    if(threadID < 2*borderSizeReal){
    int chunkNum=0; 

     if(threadID < borderSizeReal){
       chunkNum = threadID/chunkSize;
        //diff[chunkNum] = CAL_TRUE;
      }
     else{
        chunkNum = ((rows-1)*columns+threadID)/chunkSize;
       // diff[chunkNum] = CAL_TRUE;
     }

    }*/
    

}

inline void mergeBuffers3D(CALint i,
	              __global CALbyte* A,
                          CALint j,
                __global CALbyte* B){

	B[j] = B[j] || A[i];
}


__kernel void calclMergeFlags3D(
  CALint rows,
  CALint columns,
  CALint slices,
  CALint borderSize,
  __global CALbyte* mergeflagsBuffer,
  __global CALbyte* flagsReal){

  int threadID = get_global_id (0);
  const int borderSizeReal= borderSize*columns*rows;
  const int realSize = columns*rows*slices;

  

  if(threadID < 2*borderSizeReal){
    int chunkNum=0; 

    if(threadID < borderSizeReal){
      //chunkNum = threadID/chunkSize;
      mergeBuffers3D(threadID,mergeflagsBuffer,threadID,flagsReal);
       
    }
    else{
      // chunkNum = ((rows-1)*columns+threadID)/chunkSize;
         mergeBuffers3D(threadID,
             mergeflagsBuffer,
             threadID-borderSizeReal,
             flagsReal+(realSize-(borderSizeReal))
             );           
    }


    

  }
    
}


/*
__kernel void calclMergeFlags2D(
	CALint rows,
	CALint columns,
	CALint borderSize,
	__global CALbyte* mergeflagsBuffer,
	__global CALbyte* flags) 
{
	int threadID = get_global_id (0);
	const int borderSizeReal= borderSize*columns;
	const int fullSize = columns*(rows+borderSize*2);

	if(threadID < 2*borderSizeReal){

		if(threadID < borderSizeReal){
			mergeBuffers(threadID,mergeflagsBuffer,flags+borderSizeReal);
		}
		else{
			mergeBuffers(threadID,
						 mergeflagsBuffer+borderSizeReal,
						 flags+(fullSize-(borderSizeReal*2))
						 );
		}

    }
    
}
*/

