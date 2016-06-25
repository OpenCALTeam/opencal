#ifndef OPENCALTIME
#define OPENCALTIME

#include <sys/time.h>
#include <stdio.h>

struct OpenCALTime{

    struct timeval tmStart;

    struct timeval tmEnd;
};

void startTime(struct OpenCALTime * opencalTime){
    gettimeofday(&opencalTime->tmStart, NULL);
}

void endTime(struct OpenCALTime * opencalTime){
    gettimeofday(&opencalTime->tmEnd, NULL);
    unsigned long long seconds =(opencalTime->tmEnd.tv_sec - opencalTime->tmStart.tv_sec) ;
    unsigned long long milliseconds = (opencalTime->tmEnd.tv_usec - opencalTime->tmStart.tv_usec) / 1000;
    unsigned long long totalMilliseconds =1000*seconds + milliseconds;
    int totalSeconds =(int)totalMilliseconds/1000;
    int totalMinutes =(int)totalSeconds/60;
    totalSeconds =(int)totalSeconds%60;
    int totalMilliseconds2 =(int)totalMilliseconds%1000;
    printf("%d:%d.%d;",totalMinutes,totalSeconds,totalMilliseconds2);
    printf("%d;",totalMilliseconds);
    printf("%d\n",seconds);

}


#endif
