#ifndef GRAPH_H
#define	GRAPH_H

#include "utils.h"

#include <fstream>
#include <iostream>

struct Graph
{
    Graph();
    virtual ~Graph();
    friend fstream& operator >> (fstream& in, Graph& e);
    friend ostream& operator << (ostream& out, Graph& e);
    int* operator[](int k) ;
    
    int size;
    int* matrix;
};

fstream& operator >> (fstream& in, Graph& e);
#endif  /* GRAPH_H */

