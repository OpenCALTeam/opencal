#ifndef forestInsect_h
#define forestInsect_h

#include <OpenCAL/cal2D.h>
#include <OpenCAL/cal2DIO.h>
#include <OpenCAL/cal2DRun.h>
#include <OpenCAL-GL/calgl2D.h>
#include <OpenCAL-GL/calgl2DWindow.h>
#include <stdlib.h>

#define TERRAIN_PATH "./data/terrain.txt"
#define PINE_FOREST_PATH "./data/pineForest.txt"
#define OTHER_TREE "./data/otherTrees.txt"
#define PINE_AGE_PATH "./data/pineAges.txt"
#define PINE_DIAMETER_PATH "./data/pineDiameter.txt"
#define FEMALE_BETTLE_PATH "./data/femaleBettles.txt"
#define MALE_BETTLE_PATH "./data/maleBettles.txt"

enum WIND {
	NO_WIND = 0,
	NORTH_WIND,
	EAST_WIND,
	SOUTH_WIND,
	WEAST_WIND
};

enum MOVEMENT {
	MOVEMENT_NONE = 0,
	MOVEMENT_UP,
	MOVEMENT_UP_RIGHT,
	MOVEMENT_RIGHT,
	MOVEMENT_DOWN_RIGHT,
	MOVEMENT_DOWN,
	MOVEMENT_DOWN_LEFT,
	MOVEMENT_LEFT,
	MOVEMENT_UP_LEFT
};

//cadef and rundef
extern struct CALModel2D* forestInsect;
extern struct CALRun2D* forestInsectSimulation;

struct forestInsectSubstates {
	struct CALSubstate2Dr *terrain;
	struct CALSubstate2Dr *pineForest;		// Contains pine height
	struct CALSubstate2Dr *otherTrees;

	struct CALSubstate2Dr *pineHealth;		// Default is 100
	struct CALSubstate2Dr *pineAge;
	struct CALSubstate2Dr *pineDiameter;
	struct CALSubstate2Dr *pineSusceptibility; // Depends by DBH, age, previously attacked tree
	struct CALSubstate2Dr *femaleBettle;
	struct CALSubstate2Dr *femalePremetureBettle;
	struct CALSubstate2Dr *maleBettle;
	struct CALSubstate2Dr *malePremetureBettle;
	struct CALSubstate2Dr *totalFemaleBettle;	// It's the sum of the total female bettle
	struct CALSubstate2Dr *totalMaleBettle;		// It's the sum of the total male bettle
	struct CALSubstate2Dr *boolFemaleBettle;	// >1 if female bettle are presents, 0 if not
	struct CALSubstate2Dr *boolMaleBettle;		// >1 if male bettle are presents, 0 if not
	struct CALSubstate2Dr *eggs;				// Initialised with 0
	struct CALSubstate2Dr *movementFemale;		// Used for calculate the next cell where to move female
	struct CALSubstate2Dr *movementMale;		// Used for calculate the next cell where to move male
};

struct forestInsectParameters {
	enum WIND windType;
	CALreal winterModerateMortality;
	CALreal winterSevereMortality;
	CALreal springMortality;
	CALreal summerMortality;
	CALreal autumnMortality;
	CALint treesholdWheaterEffects;
	CALint startDay;
	CALint currentDay;
	CALint dayForDeposingEggs;
	CALint dayForUnlockingEggs;
	CALreal eggsDeposedByBettle;
	CALint dayToChangeTreeForBettle;
	CALreal energyDecreasedByBettles;
	CALreal energyIncreasedByYoungTrees;
	CALreal energyIncreasedByAdultTrees;
	CALreal energyIncreasedByOldTrees;
	CALreal maximumTreeEnergy;
	CALreal minimumTreeEnergy;
	CALint ageForAdultTrees;
	CALint ageForOldTrees;
	CALreal incrementHeightForTree;
	CALreal incrementDiameterForTree;
	CALreal minimumDiameter;
	CALreal maximumDiameter;
	CALreal incrementSusceptibilityForAge;
};

extern struct forestInsectSubstates Q;
extern struct forestInsectParameters P;

void forestInsectCADef();
void forestInsectLoadConfig();
void forestInsectExit(void);

// Elementary transition function
void forestInsectUpdateDay(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateAgeTree(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateHeightTree(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateDiameterTree(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectApplyWheaterEffects(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectEggsDeposing(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdatePrematureBettles(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectEggsUnlocking(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateTreeEnergy(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectCalculateTreeSusceptibility(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectMoveBettle(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateTotalBettle(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectUpdateMovement(struct CALModel2D* forestInsect, CALint i, CALint j);
void forestInsectClearMovement(struct CALModel2D* forestInsect, CALint i, CALint j);

// Utility functions
void forestInsectCalculateTotalBettle();
CALint getNextNeighboorhoodWithWind_Female(struct CALModel2D* forestInsect, CALint i, CALint j);
CALint getNextNeighboorhoodWithWind_Male(struct CALModel2D* forestInsect, CALint i, CALint j);

#endif
