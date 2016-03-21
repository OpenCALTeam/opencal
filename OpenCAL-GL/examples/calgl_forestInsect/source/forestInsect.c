#include "forestInsect.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

//-----------------------------------------------------------------------
//	   ForestInsect cellular automaton definition section
//-----------------------------------------------------------------------

#define ROWS 177
#define COLUMNS 177
#define STEPS 3000

//cadef and rundef
struct CALModel2D* forestInsect;			//the cellular automaton
struct forestInsectSubstates Q;				//the substates
struct forestInsectParameters P;			//the parameters
struct CALRun2D* forestInsectSimulation;	//the simulartion run

#define WIND_DIMENSION 5
CALint northWind[WIND_DIMENSION] = {2,3,4,6,7};
CALint eastWind[WIND_DIMENSION] = {1,2,4,5,6};
CALint southWind[WIND_DIMENSION] = {1,2,3,5,8};
CALint weastWind[WIND_DIMENSION] = {1,3,4,7,8};

//------------------------------------------------------------------------------
//					ForestInsect transition function
//------------------------------------------------------------------------------

void forestInsectUpdateDay(struct CALModel2D* forestInsect, CALint i, CALint j){
	static CALint oldValue = 0;
	CALint tmpDay = 0;
	CALint month = 1;
	char* windStrings[] = {"None", "North", "East", "South", "Weast"};

	if (oldValue != forestInsectSimulation->step){
		oldValue = forestInsectSimulation->step;
		tmpDay = forestInsectSimulation->step % 360;
		tmpDay = tmpDay == 0 ? 360 : tmpDay;
		P.currentDay = (tmpDay + P.startDay) % 360;
		P.currentDay = P.currentDay == 0 ? 360 : P.currentDay;

		printf("Day: %d\n", P.currentDay);
		month = 1 + P.currentDay / 30;
		month = month > 12 ? 12 : month;
		printf("Month: %d\n", month);

		// Wind printing
		printf("Type wind: %s", windStrings[P.windType]);
	}
}
void forestInsectUpdateAgeTree(struct CALModel2D* forestInsect, CALint i, CALint j){
	if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) > P.minimumTreeEnergy && P.currentDay == 360){
		calSet2Dr(forestInsect, Q.pineAge, i, j, calGet2Dr(forestInsect, Q.pineAge, i, j) + 1);
	}
}
void forestInsectUpdateHeightTree(struct CALModel2D* forestInsect, CALint i, CALint j){
	if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) > P.minimumTreeEnergy && P.currentDay == 360){
		calSet2Dr(forestInsect, Q.pineForest, i, j, calGet2Dr(forestInsect, Q.pineForest, i, j) + P.incrementHeightForTree);
	}
}
void forestInsectUpdateDiameterTree(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal increment = 0.0;
	if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) > P.minimumTreeEnergy && P.currentDay == 360){
		increment = calGet2Dr(forestInsect, Q.pineDiameter, i, j) + P.incrementDiameterForTree;
		increment = increment > P.maximumDiameter ? P.maximumDiameter : increment;
		calSet2Dr(forestInsect, Q.pineDiameter, i, j, increment);
	}
}
void forestInsectApplyWheaterEffects(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal prematureFemale = 0.0;
	CALreal female = 0.0;
	CALreal prematureMale = 0.0;
	CALreal male = 0.0;

	if ((rand() % 100 > P.treesholdWheaterEffects)){
		return;
	}

	prematureFemale = calGet2Dr(forestInsect, Q.femalePremetureBettle, i, j);
	female = calGet2Dr(forestInsect, Q.femaleBettle, i, j);
	prematureMale = calGet2Dr(forestInsect, Q.malePremetureBettle, i, j);
	male = calGet2Dr(forestInsect, Q.maleBettle, i, j);

	if (P.currentDay <= 90){ // Winter
		if (rand() % 2 == 0){ // Moderate winter
			prematureFemale *= (1.0f - P.winterModerateMortality);
			female *= (1.0f - P.winterModerateMortality);
			prematureMale *= (1.0f - P.winterModerateMortality);
			male *= (1.0f - P.winterModerateMortality);
		}
		else { // Severe winter
			prematureFemale *= (1.0f - P.winterSevereMortality);
			female *= (1.0f - P.winterSevereMortality);
			prematureMale *= (1.0f - P.winterSevereMortality);
			male *= (1.0f - P.winterSevereMortality);
		}
	}
	else if (P.currentDay <= 180){ // Spring
		prematureFemale *= (1.0f - P.springMortality);
		female *= (1.0f - P.springMortality);
		prematureMale *= (1.0f - P.springMortality);
		male *= (1.0f - P.springMortality);
	}
	else if (P.currentDay <= 270){ // Summer
		prematureFemale *= (1.0f - P.summerMortality);
		female *= (1.0f - P.summerMortality);
		prematureMale *= (1.0f - P.summerMortality);
		male *= (1.0f - P.summerMortality);
	}
	else { // Autumn
		prematureFemale *= (1.0f - P.autumnMortality);
		female *= (1.0f - P.autumnMortality);
		prematureMale *= (1.0f - P.autumnMortality);
		male *= (1.0f - P.autumnMortality);
	}

	calSet2Dr(forestInsect, Q.femalePremetureBettle, i, j, prematureFemale);
	calSet2Dr(forestInsect, Q.femaleBettle, i, j, female);
	calSet2Dr(forestInsect, Q.malePremetureBettle, i, j, prematureMale);
	calSet2Dr(forestInsect, Q.maleBettle, i, j, male);
}
void forestInsectEggsDeposing(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal eggsToDepose = 0.0f;

	// Check if there are male?
	if (P.currentDay == P.dayForDeposingEggs){
		eggsToDepose = calGet2Dr(forestInsect, Q.femaleBettle, i, j)*P.eggsDeposedByBettle;
		calSet2Dr(forestInsect, Q.eggs, i, j, eggsToDepose);
	}
}
void forestInsectUpdatePrematureBettles(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal bettleThatPassingFromYoungToAdult = 0.0f;

	if (P.currentDay == P.dayForUnlockingEggs){
		// Female
		bettleThatPassingFromYoungToAdult = calGet2Dr(forestInsect, Q.femalePremetureBettle, i, j);
		// Adding old + new bettle and update cell
		calSet2Dr(forestInsect, Q.femaleBettle, i, j, calGet2Dr(forestInsect, Q.femaleBettle, i, j) + bettleThatPassingFromYoungToAdult);
		calSet2Dr(forestInsect, Q.femalePremetureBettle, i, j, 0.0);
		// Male
		bettleThatPassingFromYoungToAdult = calGet2Dr(forestInsect, Q.malePremetureBettle, i, j);
		// Adding old + new bettle and update cell
		calSet2Dr(forestInsect, Q.maleBettle, i, j, calGet2Dr(forestInsect, Q.maleBettle, i, j) + bettleThatPassingFromYoungToAdult);
		calSet2Dr(forestInsect, Q.malePremetureBettle, i, j, 0.0);
	}
}
void forestInsectEggsUnlocking(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal newBettles = 0.0;
	CALreal randType = 0.0;
	CALreal newFemale = 0.1;	// A new 10% is female
	CALreal newMale = 0.1;	// A new 10% is male

	if (P.currentDay == P.dayForUnlockingEggs){
		// Update substate eggs
		newBettles = calGet2Dr(forestInsect, Q.eggs, i, j);
		calSet2Dr(forestInsect, Q.eggs, i, j, 0.0);
		// The remaining 80% is divided by two categories
		randType = ((rand() % 80) / 80.0);
		newFemale += randType;
		newMale += (0.8 - randType);
		newFemale *= newBettles;
		newMale *= newBettles;
		// Set new female bettles
		calSet2Dr(forestInsect, Q.femalePremetureBettle, i, j, newFemale);
		// Set new male bettles
		calSet2Dr(forestInsect, Q.malePremetureBettle, i, j, newMale);
	}
}
void forestInsectUpdateTotalBettle(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal totalFemaleBettle = 0.0, totalMaleBettle = 0.0;

	totalFemaleBettle += calGet2Dr(forestInsect, Q.femaleBettle, i, j);
	totalFemaleBettle += calGet2Dr(forestInsect, Q.femalePremetureBettle, i, j);
	totalMaleBettle += calGet2Dr(forestInsect, Q.maleBettle, i, j);
	totalMaleBettle += calGet2Dr(forestInsect, Q.malePremetureBettle, i, j);

	if (totalFemaleBettle > 0.0){
		calSet2Dr(forestInsect, Q.boolFemaleBettle, i, j, 1.0 + calGet2Dr(forestInsect, Q.pineForest, i, j));
	}
	else {
		calSet2Dr(forestInsect, Q.boolFemaleBettle, i, j, calGet2Dr(forestInsect, Q.pineForest, i, j));
	}

	if (totalMaleBettle > 0.0){
		calSet2Dr(forestInsect, Q.boolMaleBettle, i, j, 1.0 + calGet2Dr(forestInsect, Q.pineForest, i, j));
	}
	else {
		calSet2Dr(forestInsect, Q.boolMaleBettle, i, j, calGet2Dr(forestInsect, Q.pineForest, i, j));
	}

	totalFemaleBettle = totalFemaleBettle == 0.0 ? 0.1 : totalFemaleBettle;
	calSet2Dr(forestInsect, Q.totalFemaleBettle, i, j, totalFemaleBettle);
	totalMaleBettle = totalMaleBettle == 0.0 ? 0.1 : totalMaleBettle;
	calSet2Dr(forestInsect, Q.totalMaleBettle, i, j, totalMaleBettle);
}
void forestInsectUpdateTreeEnergy(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal minus = 0.0;
	CALreal plus = 0.0;
	CALreal pineAge = 0.0;
	CALreal finalEnergy = 0.0;

	if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) > P.minimumTreeEnergy)
	{
		if(calGet2Dr(forestInsect, Q.totalFemaleBettle, i, j)>1.0){
			minus += calGet2Dr(forestInsect, Q.totalFemaleBettle, i, j)*P.energyDecreasedByBettles;
		}
		if(calGet2Dr(forestInsect, Q.totalMaleBettle, i, j)>1.0){
			minus += calGet2Dr(forestInsect, Q.totalMaleBettle, i, j)*P.energyDecreasedByBettles;
		}
		pineAge = calGet2Dr(forestInsect, Q.pineAge, i, j);
		if (pineAge < P.ageForAdultTrees){
			plus = pineAge*P.energyIncreasedByYoungTrees;
		}
		else if (pineAge<P.ageForOldTrees){
			plus = pineAge*P.energyIncreasedByAdultTrees;
		}
		else {
			plus = pineAge*P.energyIncreasedByOldTrees;
		}
		finalEnergy = calGet2Dr(forestInsect, Q.pineHealth, i, j) + plus - minus;
		finalEnergy = finalEnergy > P.maximumTreeEnergy ? P.maximumTreeEnergy : finalEnergy;
		finalEnergy = finalEnergy >= P.minimumTreeEnergy ? finalEnergy : P.minimumTreeEnergy;
		calSet2Dr(forestInsect, Q.pineHealth, i, j, finalEnergy);
	}
}
void forestInsectCalculateTreeSusceptibility(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALreal diameter = 0.0;
	CALreal age = 0.0;
	CALreal health = calGet2Dr(forestInsect, Q.pineHealth, i, j);
	CALreal susceptibility = 0.0;

	if (health > P.minimumTreeEnergy){ // If there is an alive pine
		diameter = calGet2Dr(forestInsect, Q.pineDiameter, i, j);
		age = calGet2Dr(forestInsect, Q.pineAge, i, j);

		susceptibility += ((diameter*P.minimumDiameter) / (P.maximumDiameter - P.minimumDiameter))*100.0; // diameter
		susceptibility += age*P.incrementSusceptibilityForAge; // age
		susceptibility += (100.0 - health); // health
	}
	calSet2Dr(forestInsect, Q.pineSusceptibility, i, j, susceptibility);
}
void forestInsectMoveBettle(struct CALModel2D* forestInsect, CALint i, CALint j){
	CALint nextNeighboorhood = 0;

#pragma region FemaleMovement
	nextNeighboorhood = getNextNeighboorhoodWithWind_Female(forestInsect, i, j);

	if(nextNeighboorhood!=0){
		switch (nextNeighboorhood)
		{
		case 1:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_UP);
			break;
		case 2:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_LEFT);
			break;
		case 3:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_RIGHT);
			break;
		case 4:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_DOWN);
			break;
		case 5:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_UP_LEFT);
			break;
		case 6:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_DOWN_LEFT);
			break;
		case 7:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_DOWN_RIGHT);
			break;
		case 8:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_UP_RIGHT);
			break;
		default:
			calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_NONE);
			break;
		}
	}
	else
	{
		calSet2Dr(forestInsect, Q.movementFemale, i, j, MOVEMENT_NONE);
	}
#pragma endregion

#pragma region MaleMovement
	nextNeighboorhood = getNextNeighboorhoodWithWind_Male(forestInsect, i, j);

	if(nextNeighboorhood!=0){
		switch (nextNeighboorhood)
		{
		case 1:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_UP);
			break;
		case 2:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_LEFT);
			break;
		case 3:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_RIGHT);
			break;
		case 4:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_DOWN);
			break;
		case 5:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_UP_LEFT);
			break;
		case 6:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_DOWN_LEFT);
			break;
		case 7:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_DOWN_RIGHT);
			break;
		case 8:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_UP_RIGHT);
			break;
		default:
			calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_NONE);
			break;
		}
	}
	else
	{
		calSet2Dr(forestInsect, Q.movementMale, i, j, MOVEMENT_NONE);
	}
#pragma endregion

}
CALint getNextNeighboorhoodWithWind_Female(struct CALModel2D* forestInsect, CALint i, CALint j){
	//	 5 | 1 | 8
	//	---|---|---
	//	 2 | 0 | 3
	//	---|---|---
	//	 6 | 4 | 7
	CALint n = 0, notFound = 1;
	CALint indexMaxSusceptibility = 0;
	CALreal valueMaxSusceptibility = 0.0;

	switch(P.windType){
	case NORTH_WIND:
#pragma region NORTH_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, northWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, northWind[n]) > P.minimumTreeEnergy)
				{ // If in the n-neighboor there is  a tree
					if (calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, northWind[n]) > valueMaxSusceptibility)
					{
						indexMaxSusceptibility = northWind[n];
						valueMaxSusceptibility = calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, northWind[n]);
					}
				}
			}
			if (indexMaxSusceptibility == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, northWind[n]) > 0.0){
						indexMaxSusceptibility = northWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case EAST_WIND:
#pragma region EAST_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, eastWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, eastWind[n]) > P.minimumTreeEnergy)
				{ // If in the n-neighboor there is  a tree
					if (calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, eastWind[n]) > valueMaxSusceptibility)
					{
						indexMaxSusceptibility = eastWind[n];
						valueMaxSusceptibility = calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, eastWind[n]);
					}
				}
			}
			if (indexMaxSusceptibility == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, eastWind[n]) > 0.0){
						indexMaxSusceptibility = eastWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case SOUTH_WIND:
#pragma region SOUTH_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, southWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, southWind[n]) > P.minimumTreeEnergy)
				{ // If in the n-neighboor there is  a tree
					if (calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, southWind[n]) > valueMaxSusceptibility)
					{
						indexMaxSusceptibility = southWind[n];
						valueMaxSusceptibility = calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, southWind[n]);
					}
				}
			}
			if (indexMaxSusceptibility == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, southWind[n]) > 0.0){
						indexMaxSusceptibility = southWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case WEAST_WIND:
#pragma region WEAST_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, weastWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, weastWind[n]) > P.minimumTreeEnergy)
				{ // If in the n-neighboor there is  a tree
					if (calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, weastWind[n]) > valueMaxSusceptibility)
					{
						indexMaxSusceptibility = weastWind[n];
						valueMaxSusceptibility = calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, weastWind[n]);
					}
				}
			}
			if (indexMaxSusceptibility == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, weastWind[n]) > 0.0){
						indexMaxSusceptibility = weastWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	default:
#pragma region NO_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 1; n <= 8; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, n) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, n) > P.minimumTreeEnergy)
				{ // If in the n-neighboor there is  a tree
					if (calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, n) > valueMaxSusceptibility)
					{
						indexMaxSusceptibility = n;
						valueMaxSusceptibility = calGetX2Dr(forestInsect, Q.pineSusceptibility, i, j, n);
					}
				}
			}
			if (indexMaxSusceptibility == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = 1 + rand() % 8;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, n) > 0.0){
						indexMaxSusceptibility = n;
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	}

	return indexMaxSusceptibility;
}
CALint getNextNeighboorhoodWithWind_Male(struct CALModel2D* forestInsect, CALint i, CALint j){
	//	 5 | 1 | 8
	//	---|---|---
	//	 2 | 0 | 3
	//	---|---|---
	//	 6 | 4 | 7
	CALint n = 0, notFound = 1;
	CALint indexFemale = 0;
	CALreal valueFemale = 0.0, valueTmpFemale = 0.0;

	switch(P.windType){
	case NORTH_WIND:
#pragma region NORTH_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, northWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, northWind[n]) > P.minimumTreeEnergy)
				{
					valueTmpFemale = calGetX2Dr(forestInsect, Q.femaleBettle, i, j, northWind[n]) + calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, northWind[n]);
					if (valueTmpFemale > valueFemale)
					{
						indexFemale = northWind[n];
						valueFemale = valueTmpFemale;
					}
				}
			}
			if (indexFemale == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, northWind[n]) > 0.0){
						indexFemale = northWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case EAST_WIND:
#pragma region EAST_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, eastWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, eastWind[n]) > P.minimumTreeEnergy)
				{
					valueTmpFemale = calGetX2Dr(forestInsect, Q.femaleBettle, i, j, eastWind[n]) + calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, eastWind[n]);
					if (valueTmpFemale > valueFemale)
					{
						indexFemale = eastWind[n];
						valueFemale = valueTmpFemale;
					}
				}
			}
			if (indexFemale == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, eastWind[n]) > 0.0){
						indexFemale = eastWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case SOUTH_WIND:
#pragma region SOUTH_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, southWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, southWind[n]) > P.minimumTreeEnergy)
				{
					valueTmpFemale = calGetX2Dr(forestInsect, Q.femaleBettle, i, j, southWind[n]) + calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, southWind[n]);
					if (valueTmpFemale > valueFemale)
					{
						indexFemale = southWind[n];
						valueFemale = valueTmpFemale;
					}
				}
			}
			if (indexFemale == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, southWind[n]) > 0.0){
						indexFemale = southWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	case WEAST_WIND:
#pragma region WEAST_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 0; n < WIND_DIMENSION; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, weastWind[n]) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, weastWind[n]) > P.minimumTreeEnergy)
				{
					valueTmpFemale = calGetX2Dr(forestInsect, Q.femaleBettle, i, j, weastWind[n]) + calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, weastWind[n]);
					if (valueTmpFemale > valueFemale)
					{
						indexFemale = weastWind[n];
						valueFemale = valueTmpFemale;
					}
				}
			}
			if (indexFemale == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = rand() % WIND_DIMENSION;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, weastWind[n]) > 0.0){
						indexFemale = weastWind[n];
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	default:
#pragma region NO_WIND
		if (calGet2Dr(forestInsect, Q.pineForest, i, j) > 0.0 && calGet2Dr(forestInsect, Q.pineHealth, i, j) <= P.minimumTreeEnergy)
		{
			for (n = 1; n <= 8; n++)
			{
				if (calGetX2Dr(forestInsect, Q.pineForest, i, j, n) > 0.0 && calGetX2Dr(forestInsect, Q.pineHealth, i, j, n) > P.minimumTreeEnergy)
				{
					valueTmpFemale = calGetX2Dr(forestInsect, Q.femaleBettle, i, j, n) + calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, n);
					if (valueTmpFemale > valueFemale)
					{
						indexFemale = n;
						valueFemale = valueTmpFemale;
					}
				}
			}
			if (indexFemale == 0){ // Random choose
				notFound = 1;
				while (notFound){
					n = 1 + rand() % 8;
					// If in the n-neighboor there is  a tree and it is alive
					if (calGetX2Dr(forestInsect, Q.pineForest, i, j, n) > 0.0){
						indexFemale = n;
						notFound = 0;
					}
				}
			}
		}
#pragma endregion
		break;
	}

	return indexFemale;
}
void forestInsectUpdateMovement(struct CALModel2D* forestInsect, CALint i, CALint j)
{
	CALreal prematureFemale = 0.0;
	CALreal female = 0.0;
	CALreal prematureMale = 0.0;
	CALreal male = 0.0;

#pragma region FemaleMovement
	prematureFemale = 0.0;
	female = 0.0;

	if (calGet2Dr(forestInsect, Q.movementFemale, i, j) == MOVEMENT_NONE)
	{
		prematureFemale += calGet2Dr(forestInsect, Q.femalePremetureBettle, i, j);
		female += calGet2Dr(forestInsect, Q.femaleBettle, i, j);
	}

	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 1) == MOVEMENT_DOWN){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 1);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 1);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 2) == MOVEMENT_RIGHT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 2);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 2);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 3) == MOVEMENT_LEFT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 3);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 3);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 4) == MOVEMENT_UP){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 4);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 4);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 5) == MOVEMENT_DOWN_RIGHT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 5);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 5);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 6) == MOVEMENT_UP_RIGHT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 6);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 6);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 7) == MOVEMENT_UP_LEFT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 7);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 7);
	}
	if (calGetX2Dr(forestInsect, Q.movementFemale, i, j, 8) == MOVEMENT_DOWN_LEFT){
		prematureFemale += calGetX2Dr(forestInsect, Q.femalePremetureBettle, i, j, 8);
		female += calGetX2Dr(forestInsect, Q.femaleBettle, i, j, 8);
	}

	calSet2Dr(forestInsect, Q.femalePremetureBettle, i, j, prematureFemale);
	calSet2Dr(forestInsect, Q.femaleBettle, i, j, female);
#pragma endregion

#pragma region MaleMovement
	prematureMale = 0.0;
	male = 0.0;

	if (calGet2Dr(forestInsect, Q.movementMale, i, j) == MOVEMENT_NONE)
	{
		prematureMale += calGet2Dr(forestInsect, Q.malePremetureBettle, i, j);
		male += calGet2Dr(forestInsect, Q.maleBettle, i, j);
	}

	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 1) == MOVEMENT_DOWN){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 1);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 1);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 2) == MOVEMENT_RIGHT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 2);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 2);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 3) == MOVEMENT_LEFT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 3);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 3);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 4) == MOVEMENT_UP){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 4);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 4);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 5) == MOVEMENT_DOWN_RIGHT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 5);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 5);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 6) == MOVEMENT_UP_RIGHT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 6);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 6);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 7) == MOVEMENT_UP_LEFT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 7);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 7);
	}
	if (calGetX2Dr(forestInsect, Q.movementMale, i, j, 8) == MOVEMENT_DOWN_LEFT){
		prematureMale += calGetX2Dr(forestInsect, Q.malePremetureBettle, i, j, 8);
		male += calGetX2Dr(forestInsect, Q.maleBettle, i, j, 8);
	}

	calSet2Dr(forestInsect, Q.malePremetureBettle, i, j, prematureMale);
	calSet2Dr(forestInsect, Q.maleBettle, i, j, male);
#pragma endregion

}
void forestInsectClearMovement(struct CALModel2D* forestInsect, CALint i, CALint j)
{
	calSet2Dr(forestInsect, Q.movementFemale, i,j, MOVEMENT_NONE);
	calSet2Dr(forestInsect, Q.movementMale, i,j, MOVEMENT_NONE);
}

//------------------------------------------------------------------------------
//					ForestInsect simulation functions
//------------------------------------------------------------------------------

void forestInsectSimulationInit(struct CALModel2D* forestInsect)
{

}

void forestInsectSteering(struct CALModel2D* forestInsect)
{

}

CALbyte forestInsectSimulationStopCondition(struct CALModel2D* forestInsect)
{
	if (forestInsectSimulation->step >= STEPS)
		return CAL_TRUE;
	return CAL_FALSE;
}

//------------------------------------------------------------------------------
//					ForestInsect CADef and runDef
//------------------------------------------------------------------------------

void forestInsectCADef() {
	struct CALGLDrawModel2D* drawModel = NULL;

	srand((unsigned int)time(0));

	// Setting parameter
	P.windType = NO_WIND;
	P.winterModerateMortality = 0.8f;	// 80%
	P.winterSevereMortality = 0.9f;		// 90%
	P.springMortality = 0.15f;	// 15%
	P.summerMortality = 0.1f;	// 10%
	P.autumnMortality = 0.4f;	//	40%
	P.treesholdWheaterEffects = 10;
	P.startDay = 1;
	P.currentDay = 1;
	P.dayForDeposingEggs = 150;			// In 150 year's day bettles deposes eggs
	P.dayForUnlockingEggs = 210;		// In 210 year's day bettles' eggs are unlocking
	P.eggsDeposedByBettle = 1000;		// Eggs deposed by one bettle
	P.dayToChangeTreeForBettle = 250;	// Day for bettles to change tree
	P.energyDecreasedByBettles = 10.0f;
	P.energyIncreasedByYoungTrees = 0.010f;
	P.energyIncreasedByAdultTrees = 0.018f;
	P.energyIncreasedByOldTrees = 0.005f;
	P.maximumTreeEnergy = 100.0;
	P.minimumTreeEnergy = 0.9;
	P.ageForAdultTrees = 3;
	P.ageForOldTrees = 7;
	P.incrementHeightForTree = 0.1f;
	P.incrementDiameterForTree = 0.1f;
	P.minimumDiameter = 1.0f;
	P.maximumDiameter = 4.0f;
	P.incrementSusceptibilityForAge = 0.2f;

	//cadef and rundef
	forestInsect = calCADef2D(ROWS, COLUMNS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_FLAT, CAL_NO_OPT);
	forestInsectSimulation = calRunDef2D(forestInsect, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);
	//add transition function's elementary processes
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateDay);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateAgeTree);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateHeightTree);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateDiameterTree);
	calAddElementaryProcess2D(forestInsect, forestInsectEggsDeposing);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdatePrematureBettles);
	calAddElementaryProcess2D(forestInsect, forestInsectEggsUnlocking);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateTotalBettle);
	calAddElementaryProcess2D(forestInsect, forestInsectCalculateTreeSusceptibility);
	calAddElementaryProcess2D(forestInsect, forestInsectMoveBettle);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateMovement);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateTotalBettle);
	calAddElementaryProcess2D(forestInsect, forestInsectClearMovement);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateTreeEnergy);
	calAddElementaryProcess2D(forestInsect, forestInsectApplyWheaterEffects);
	calAddElementaryProcess2D(forestInsect, forestInsectUpdateTotalBettle);
	//simulation run setup
	calRunAddInitFunc2D(forestInsectSimulation, forestInsectSimulationInit);
	calRunAddSteeringFunc2D(forestInsectSimulation, forestInsectSteering);
	calRunAddStopConditionFunc2D(forestInsectSimulation, forestInsectSimulationStopCondition);
	//add substates
	Q.terrain = calAddSubstate2Dr(forestInsect);
	Q.pineForest = calAddSubstate2Dr(forestInsect);
	Q.otherTrees = calAddSubstate2Dr(forestInsect);
	Q.pineHealth = calAddSubstate2Dr(forestInsect);
	Q.pineSusceptibility = calAddSubstate2Dr(forestInsect);
	Q.pineAge = calAddSubstate2Dr(forestInsect);
	Q.pineDiameter = calAddSubstate2Dr(forestInsect);
	Q.femaleBettle = calAddSubstate2Dr(forestInsect);
	Q.femalePremetureBettle = calAddSubstate2Dr(forestInsect);
	Q.maleBettle = calAddSubstate2Dr(forestInsect);
	Q.malePremetureBettle = calAddSubstate2Dr(forestInsect);
	Q.totalFemaleBettle = calAddSubstate2Dr(forestInsect);
	Q.totalMaleBettle = calAddSubstate2Dr(forestInsect);
	Q.boolFemaleBettle = calAddSubstate2Dr(forestInsect);
	Q.boolMaleBettle = calAddSubstate2Dr(forestInsect);
	Q.eggs = calAddSubstate2Dr(forestInsect);
	Q.movementFemale = calAddSubstate2Dr(forestInsect);
	Q.movementMale = calAddSubstate2Dr(forestInsect);

	// draw model definition
	drawModel = calglDefDrawModel2D(CALGL_DRAW_MODE_SURFACE, "ForestInsect", forestInsect, forestInsectSimulation);
	// Add Terrain
	calglAdd2Dr(drawModel, NULL, &Q.terrain, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_STATIC);
	calglColor2D(drawModel, 0.5, 0.5, 0.5, 1.0);
	calglAdd2Dr(drawModel, Q.terrain, &Q.terrain, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.terrain, &Q.terrain, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// Add Pine Forest
	calglAdd2Dr(drawModel, Q.terrain, &Q.pineForest, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.pineForest, &Q.pineHealth, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_GREEN_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.pineForest, &Q.pineForest, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// Add Other Trees
	calglAdd2Dr(drawModel, Q.terrain, &Q.otherTrees, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglColor2D(drawModel, 159.0f / 255, 76.0f / 255, 0.1f, 1.0);
	calglAdd2Dr(drawModel, Q.otherTrees, &Q.otherTrees, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_CURRENT_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.otherTrees, &Q.otherTrees, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// Add Female Bettle to drawing
	calglAdd2Dr(drawModel, Q.terrain, &Q.boolFemaleBettle, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.boolFemaleBettle, &Q.totalFemaleBettle, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_BLUE_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.boolFemaleBettle, &Q.boolFemaleBettle, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	// Add Male Bettle to drawing
	calglAdd2Dr(drawModel, Q.terrain, &Q.boolMaleBettle, CALGL_TYPE_INFO_VERTEX_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.boolMaleBettle, &Q.totalMaleBettle, CALGL_TYPE_INFO_COLOR_DATA, CALGL_TYPE_INFO_USE_RED_SCALE, CALGL_DATA_TYPE_DYNAMIC);
	calglAdd2Dr(drawModel, Q.boolMaleBettle, &Q.boolMaleBettle, CALGL_TYPE_INFO_NORMAL_DATA, CALGL_TYPE_INFO_USE_NO_COLOR, CALGL_DATA_TYPE_DYNAMIC);

	forestInsectLoadConfig();
}

//------------------------------------------------------------------------------
//					ForestInsect I/O functions
//------------------------------------------------------------------------------

void forestInsectLoadConfig()
{
	calLoadSubstate2Dr(forestInsect, Q.terrain, TERRAIN_PATH);
	calLoadSubstate2Dr(forestInsect, Q.pineForest, PINE_FOREST_PATH);
	calLoadSubstate2Dr(forestInsect, Q.otherTrees, OTHER_TREE);

	calInitSubstate2Dr(forestInsect, Q.pineHealth, P.maximumTreeEnergy);
	calLoadSubstate2Dr(forestInsect, Q.pineAge, PINE_AGE_PATH);
	calLoadSubstate2Dr(forestInsect, Q.pineDiameter, PINE_DIAMETER_PATH);
	calInitSubstate2Dr(forestInsect, Q.pineSusceptibility, 0.0);

	calLoadSubstate2Dr(forestInsect, Q.femaleBettle, FEMALE_BETTLE_PATH);
	calInitSubstate2Dr(forestInsect, Q.femalePremetureBettle, 0.0);
	calLoadSubstate2Dr(forestInsect, Q.maleBettle, MALE_BETTLE_PATH);
	calInitSubstate2Dr(forestInsect, Q.malePremetureBettle, 0.0);
	calInitSubstate2Dr(forestInsect, Q.totalFemaleBettle, 0.0);
	calInitSubstate2Dr(forestInsect, Q.totalMaleBettle, 0.0);
	calInitSubstate2Dr(forestInsect, Q.boolFemaleBettle, 0.0);
	calInitSubstate2Dr(forestInsect, Q.boolMaleBettle, 0.0);
	forestInsectCalculateTotalBettle();
	calInitSubstate2Dr(forestInsect, Q.eggs, 0.0);
	calInitSubstate2Dr(forestInsect, Q.movementFemale, MOVEMENT_NONE);
	calInitSubstate2Dr(forestInsect, Q.movementMale, MOVEMENT_NONE);

	forestInsectSimulation->init(forestInsect);
	calUpdate2D(forestInsect);
}

//------------------------------------------------------------------------------
//					ForestInsect finalization function
//------------------------------------------------------------------------------

void forestInsectExit(void)
{
	//finalizations
	calRunFinalize2D(forestInsectSimulation);
	calFinalize2D(forestInsect);
}

void forestInsectCalculateTotalBettle(){
	CALint i = 0, j = 0;
	CALreal k = 0;
	// Summ female and male
	for (i = 0; i < forestInsect->rows; i++)
	{
		for (j = 0; j<forestInsect->columns; j++)
		{
			k = 0.0;
			k = calGet2Dr(forestInsect, Q.femaleBettle, i, j);
			if (k > 0.0){
				calSet2Dr(forestInsect, Q.boolFemaleBettle, i, j, 1.0 + calGet2Dr(forestInsect, Q.pineForest, i, j));
			}
			else {
				calSet2Dr(forestInsect, Q.boolFemaleBettle, i, j, calGet2Dr(forestInsect, Q.pineForest, i, j));
				k = 0.1;
			}
			calSet2Dr(forestInsect, Q.totalFemaleBettle, i, j, k);


			k = calGet2Dr(forestInsect, Q.maleBettle, i, j);
			if (k > 0.0){
				calSet2Dr(forestInsect, Q.boolMaleBettle, i, j, 1.0 + calGet2Dr(forestInsect, Q.pineForest, i, j));
			}
			else {
				calSet2Dr(forestInsect, Q.boolMaleBettle, i, j, calGet2Dr(forestInsect, Q.pineForest, i, j));
				k = 0.1;
			}
			calSet2Dr(forestInsect, Q.totalMaleBettle, i, j, k);
		}
	}
	calUpdateSubstate2Dr(forestInsect, Q.totalFemaleBettle);
	calUpdateSubstate2Dr(forestInsect, Q.totalMaleBettle);
	calUpdateSubstate2Dr(forestInsect, Q.boolFemaleBettle);
	calUpdateSubstate2Dr(forestInsect, Q.boolMaleBettle);
}
