/*+++++++++++++++++++++++++++++++++++++++*/
/*	First Example						 */
/*+++++++++++++++++++++++++++++++++++++++*/

#include "forestInsect.h"
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>

int main(int argc, char** argv){
	calglInitViewer("Forest Insect", 4.1f, 720, 640, 0, 0, CAL_TRUE, 1);
	forestInsectCADef();
	atexit(forestInsectExit);
	calglMainLoop2D(argc, argv);
	return 0;
}
