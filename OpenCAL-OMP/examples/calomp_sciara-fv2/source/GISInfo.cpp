#include "GISInfo.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char ncols_str[]          = "ncols";
const char nrows_str[]          = "nrows";
const char xllcorner_str[]      = "xllcorner";
const char yllcorner_str[]      = "yllcorner";
const char cell_size_str[]      = "cellsize";
const char NODATA_value_str[]   = "NODATA_value";

int LeggiGISInfo(TGISInfo &gis_info, FILE* f)
{
    char str[255];
    int cont = -1;
    fpos_t position;

    //ncols
    fscanf(f,"%s",str); if (strcmp(str, ncols_str))        return GIS_FILE_GENERIC_ERROR;
    fscanf(f,"%s",str); gis_info.ncols = atoi(str);
    //nrows
    fscanf(f,"%s",str); if (strcmp(str, nrows_str))        return GIS_FILE_GENERIC_ERROR;
    fscanf(f,"%s",str); gis_info.nrows = atoi(str);
    //xll_corner
    fscanf(f,"%s",str); if (strcmp(str, xllcorner_str))    return GIS_FILE_GENERIC_ERROR;
    fscanf(f,"%s",str); gis_info.xllcorner = atof(str);
    //yll_corner
    fscanf(f,"%s",str); if (strcmp(str, yllcorner_str))    return GIS_FILE_GENERIC_ERROR;
    fscanf(f,"%s",str); gis_info.yllcorner = atof(str);
    //aphothem
    fscanf(f,"%s",str); if (strcmp(str, cell_size_str))    return GIS_FILE_TASSELATION_ERROR;
	fscanf(f,"%s",str); gis_info.cell_size = atof(str);
    //NODATA_value
    fscanf(f,"%s",str); if (strcmp(str, NODATA_value_str)) return GIS_FILE_GENERIC_ERROR;
    fscanf(f,"%s",str); gis_info.NODATA_value = atof(str);

    //verifica se il numero di elementi � consistente rispetto a ncols e nrows
    fgetpos (f, &position);
    while (!feof(f))
    {
        fscanf(f,"%s",str);
        cont ++;
    }
    fsetpos (f, &position);
    if (gis_info.ncols * gis_info.nrows != cont)
        return GIS_FILE_DIMENSION_ERROR;

    return GIS_FILE_OK;
}
//---------------------------------------------------------------------------
int VerificaGISInfo(TGISInfo gis_info, TGISInfo gis_info_morfologia)
{
    if (gis_info.ncols != gis_info_morfologia.ncols) return GIS_FILE_DIMENSION_ERROR;
    if (gis_info.nrows != gis_info_morfologia.nrows) return GIS_FILE_DIMENSION_ERROR;
	if (fabs(gis_info.xllcorner - gis_info_morfologia.xllcorner) > gis_info_morfologia.cell_size) return GIS_FILE_POSITION_ERROR;
	if (fabs(gis_info.yllcorner - gis_info_morfologia.yllcorner) > gis_info_morfologia.cell_size) return GIS_FILE_POSITION_ERROR;
	if (gis_info.cell_size != gis_info_morfologia.cell_size) return GIS_FILE_APOTHEM_ERROR;

    return GIS_FILE_OK;
}
//---------------------------------------------------------------------------
int SalvaGISInfo(const TGISInfo &gis_info, FILE* f)
{
    char str[255];

    //ncols
    fprintf(f,"%s\t\t", ncols_str);
    sprintf(str,"%d", gis_info.ncols);
    fprintf(f,"%s\n", str);
    //nrows
    fprintf(f,"%s\t\t", nrows_str);
    sprintf(str,"%d", gis_info.nrows);
    fprintf(f,"%s\n", str);
    //xllcorner
    fprintf(f,"%s\t", xllcorner_str);
    sprintf(str,"%f", gis_info.xllcorner);
    fprintf(f,"%s\n", str);
    //yllcorner
    fprintf(f,"%s\t", yllcorner_str);
    sprintf(str,"%f", gis_info.yllcorner);
    fprintf(f,"%s\n", str);
    //cell_size
    fprintf(f,"%s\t", cell_size_str);
	sprintf(str,"%f", gis_info.cell_size);
    fprintf(f,"%s\n", str);
    //NODATA_value
    fprintf(f,"%s\t", NODATA_value_str);
    sprintf(str,"%f", gis_info.NODATA_value);
    fprintf(f,"%s\n", str);

    return GIS_FILE_OK;
}
//---------------------------------------------------------------------------
void initGISInfoNODATA0(const TGISInfo &gis_info_source, TGISInfo &gis_info_dest)
{
	gis_info_dest.ncols = gis_info_source.ncols;
	gis_info_dest.nrows = gis_info_source.nrows;
	gis_info_dest.cell_size = gis_info_source.cell_size;
	gis_info_dest.NODATA_value = 0;
	gis_info_dest.xllcorner = gis_info_source.xllcorner;
	gis_info_dest.yllcorner = gis_info_source.yllcorner;
}//---------------------------------------------------------------------------
