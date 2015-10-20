/*+++++++++++++++++++++++++++++++++++++++*/
/*	First Example						 */
/*+++++++++++++++++++++++++++++++++++++++*/

#include "forestInsect.h"
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>

int main(int argc, char** argv){
	calglSetApplicationName("Forest Insect");
	calglSetCellSize(4.1f);
	calglSetWindowDimension(720, 640);
	calglSetWindowPosition(0, 0);
	calglEnableLights();
	//calglSetFixedDisplayStep(50);

	forestInsectCADef();

	calglStartProcessWindow2D(argc, argv);

	// Free Heap Memory
	forestInsectExit();
	calglDestroyGlobalSettings();
	return 0;
}


