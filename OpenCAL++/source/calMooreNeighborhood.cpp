<<<<<<< HEAD
<<<<<<< HEAD
#include <OpenCAL++/calMooreNeighborhood.h>
=======
#include <OpenCAL++11/calMooreNeighborhood.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d
=======
#include <OpenCAL++11/calMooreNeighborhood.h>
>>>>>>> e44630b317eeb506eac14bb3076f71487fe5ed2d

void CALMooreNeighborhood :: defineNeighborhood (CALModel* calModel)
{
    int n = calModel->getDimension();
    assert (n>1);
    int alphabet []= {0, -1, 1};
    int total = std::pow (3, n);

    for (int i = 0; i < total; i++)
    {
        int v = i;
        int * indexes = new int [n];
        for (int pos = 0; pos < n; pos++)
        {
            indexes[pos] = alphabet[v % 3];
            v = v / 3;
        }


     }

}
