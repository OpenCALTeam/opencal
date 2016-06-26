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

#include <OpenCAL-CL/calcl2D.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv)
{

    struct CALCLDeviceManager * calcl_device_manager;

    calcl_device_manager = calclCreateManager();
    //calclGetPlatformAndDeviceFromStdIn(calcl_device_manager, &device);
    for (int i = 0; i < calcl_device_manager->num_platforms; ++i)
        for (int j = 0; j < calcl_device_manager->num_platforms_devices[i]; ++j) {
               printf("%d;%d;%s:\n", i, j, calclGetDeviceName(calcl_device_manager->devices[i][j]));
            //printf("%d;%d \n", i, j);
        }

    // Finalizations
	calclFinalizeManager(calcl_device_manager);

	return 0;
}
