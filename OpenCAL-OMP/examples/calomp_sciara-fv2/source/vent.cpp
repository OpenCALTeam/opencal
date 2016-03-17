#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vent.h"

/*unsigned int emission_time;
vector<TEmissionRate> emission_rate;
vector<TVent> vent;*/

char time_str[]     = "[TIME]",
     vent_str[]     = "[VENT_ID]",
     emission_str[] = "[EMISSION_RATE]",
     end_str[]      = "[END]";


void initVents(int* Mv, int lx, int ly, vector<TVent>& vent)
{
    if (vent.size() > 0)
    {
        vent.clear();
        vent.resize(0);
    }
    for (int i=0; i<lx*ly; i++)
            if (Mv[i]>0)
            {
                TVent v;
                v.set_vent_id(Mv[i]);
                v.set_x(i%lx);
                v.set_y(i/lx);
                vent.push_back(v);
            }
}

void addVent(int x, int y, int vent_id, vector<TVent>& vent)
{
    TVent v;
    v.set_vent_id(vent_id);
    v.set_x(x);
    v.set_y(y);
    vent.push_back(v);
}

void removeLastVent(vector<TVent>& vent)
{
    vent.resize(vent.size()-1);
}

int loadEmissionRates(FILE *f, unsigned int& emission_time, vector<TEmissionRate>& er_vec, vector<TVent> &vent)
{
    bool ok;
    char str[255];

    if (er_vec.size() > 0)
    {
        er_vec.clear();
        er_vec.resize(0);
    }

    fscanf(f,"%s",str);
    if (!strcmp(str, time_str))
    {
        fscanf(f,"%s",str);
        if (atoi(str) < 0)
            return EMISSION_RATE_FILE_ERROR;
        emission_time = atoi(str);
    }
    else
        return EMISSION_RATE_FILE_ERROR;

    fscanf(f,"%s",str);
    do{
        if (!strcmp(str, vent_str))
        {
            TEmissionRate er;

            fscanf(f,"%s",str);
            if (atoi(str) < 0)
                return EMISSION_RATE_FILE_ERROR;

            ok = false;
            for (unsigned int i=0; i<vent.size(); i++)
                if (vent[i].vent_id() == atoi(str))
                {
                    er.set_vent_id(atoi(str));
                    ok = true;
                    break;
                }
            if (!ok)
                return EMISSION_RATE_FILE_ERROR;

            fscanf(f,"%s",str);
            if (!strcmp(str, emission_str))
            {
                fscanf(f,"%s",str);
                if (atof(str) < 0)
                    return EMISSION_RATE_FILE_ERROR;
                while(strcmp(str, vent_str))
                {
                    er.emission_rate().push_back(atof(str));
                    fscanf(f,"%s",str);
                    if (!strcmp(str, end_str))
                    {
                        er_vec.push_back(er);
                        goto done;
                    }
                    if (feof(f))
                        return EMISSION_RATE_FILE_ERROR;
                }
            }
            else
                return EMISSION_RATE_FILE_ERROR;
            er_vec.push_back(er);
        }
        else
            return EMISSION_RATE_FILE_ERROR;
    } while(!feof(f));

    done:
    for (unsigned int i=0; i<vent.size(); i++)
        for (unsigned int j=0; j<er_vec.size(); j++)
        {
            ok = false;
            if (vent[i].vent_id() == er_vec[j].vent_id())
            {
                ok = true;
                break;
            }
        }
    if  (!ok)
        return EMISSION_RATE_FILE_ERROR;

    return EMISSION_RATE_FILE_OK;
}

int loadOneEmissionRates(FILE *f, unsigned int vent_id, vector<TEmissionRate>& er_vec)
{
    //bool ok;
    char str[255];

    TEmissionRate er;
    er.set_vent_id(vent_id);
    do{
        fscanf(f,"%s",str);
        if (!strcmp(str, emission_str))
        {
            fscanf(f,"%s",str);
            if (atof(str) < 0)
                return EMISSION_RATE_FILE_ERROR;
            while(strcmp(str, end_str))
            {
                er.emission_rate().push_back(atof(str));
                fscanf(f,"%s",str);
                if (!strcmp(str, end_str))
                {
                    er_vec.push_back(er);
                    return EMISSION_RATE_FILE_OK;
                }
                if (feof(f))
                    return EMISSION_RATE_FILE_ERROR;
            }
        }
        else
            return EMISSION_RATE_FILE_ERROR;
        er_vec.push_back(er);
    } while(!feof(f));

    return EMISSION_RATE_FILE_OK;
}

int defineVents(const vector<TEmissionRate>& emission_rate, vector<TVent>& vent)
{
    for(unsigned int i=0; i<vent.size(); i++)
        if ( ! vent[i].setEmissionRate(emission_rate, vent[i].vent_id()) )
            return vent[i].vent_id();
    return 0;
}
//---------------------------------------------------------------------------
void rebuildVentsMatrix(int* Mv, int lx, int ly, vector<TVent>& vent)
{
    for (int i=0; i<lx*ly; i++)
            Mv[i]=0;
    for (unsigned int i=0; i<vent.size(); i++)
        Mv[vent[i].x() + vent[i].y()*lx] = vent[i].vent_id();

}

void saveEmissionRates(FILE *f, unsigned int emission_time, vector<TEmissionRate>& er_vec)
{
    fprintf(f, "%s\n", time_str);
    fprintf(f, "%d\n\n", emission_time);

    for (unsigned int i=0; i<er_vec.size(); i++)
    {
        fprintf(f, "%s\n", vent_str);
        fprintf(f, "%d\n\n", er_vec[i].vent_id());
        fprintf(f, "%s\n", emission_str);
        for (unsigned int j=0; j<er_vec[i].size(); j++)
            fprintf(f, "%f\n", er_vec[i][j]);
        fprintf(f, "\n");
    }

    fprintf(f, "\n%s", end_str);
}

void printEmissionRates(unsigned int emission_time, vector<TEmissionRate>& er_vec)
{
	printf("%s\n", time_str);
    printf("%d\n\n", emission_time);

    for (unsigned int i=0; i<er_vec.size(); i++)
    {
        printf("%s\n", vent_str);
        printf("%d\n\n", er_vec[i].vent_id());
        printf("%s\n", emission_str);
        for (unsigned int j=0; j<er_vec[i].size(); j++)
            printf("%f\n", er_vec[i][j]);
        printf("\n");
    }

    printf("\n%s\n", end_str);
}
//---------------------------------------------------------------------------
