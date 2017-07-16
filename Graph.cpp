#include "Graph.h"

Graph::Graph()
{
    size = 0;
    matrix = nullptr;
}

Graph::~Graph()
{
    if(size != 0)
        delete[] matrix;

    matrix = nullptr;
    size = 0;
}

int* Graph::operator[](int k)
{
    return matrix + (k * size);
}

fstream& operator >> (fstream& in, Graph& g)
{
    if(g.size != 0)
        delete[] g.matrix;

    in >> g.size;
    g.matrix = new int [g.size * g.size];

    REP(y, g.size)
        REP(x, g.size)
            in >> g[y][x];

    return in;
}

ostream& operator << (ostream& out, Graph& g)
{
    REP(y, g.size)
    {
        REP(x, g.size)
            out << g[y][x];
        out << '\n';
    }

    return out;
}
