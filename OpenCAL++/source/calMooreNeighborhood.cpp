
#include <OpenCAL++/calMooreNeighborhood.h>


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
