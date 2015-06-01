/*+++++++++++++++++++++++++++++++++++++++*/
/*	First Example						 */
/*+++++++++++++++++++++++++++++++++++++++*/

#include "forestInsect.h"
#include <calgl2D.h>
#include <calgl2DWindow.h>

int main(int argc, char** argv){
	calglSetApplicationNameGlobalSettings("Forest Insect");
	calglSetRowsAndColumnsGlobalSettings(177, 177);
	calglSetCellSizeGlobalSettings(4.1f);
	calglSetStepGlobalSettings(3000);
	calglSetWindowDimensionGlobalSettings(720, 640);
	calglSetWindowPositionGlobalSettings(0, 0);
	calglEnableLights();
	//calglSetFixedDisplayStep(50);

	forestInsectCADef();

	calglStartProcessWindow2D(argc, argv);

	// Free Heap Memory
	forestInsectExit();
	calglDestroyGlobalSettings();
	return 0;
}


