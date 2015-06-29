#ifndef GISInfo_h
#define GISInfo_h

#include <stdio.h>

#define  GIS_FILE_OK                        0
#define  GIS_FILE_GENERIC_ERROR             1
#define  GIS_FILE_TASSELATION_ERROR         2
#define  GIS_FILE_DIMENSION_ERROR           3
#define  GIS_FILE_POSITION_ERROR            4
#define  GIS_FILE_APOTHEM_ERROR             5

struct TGISInfo {
  int ncols;
  int nrows;
  double xllcorner;
  double yllcorner;
  double cell_size;
  double NODATA_value;
};

int LeggiGISInfo(TGISInfo &gis_info, FILE* f);
int VerificaGISInfo(TGISInfo gis_info, TGISInfo gis_info_morfologia);
int SalvaGISInfo(const TGISInfo &gis_info, FILE* f);
void initGISInfoNODATA0(const TGISInfo &gis_info_source, TGISInfo &gis_info_dest);

#endif