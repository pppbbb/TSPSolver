#ifndef APPLICATION_H
#define	APPLICATION_H
#include "utils.h"
#include "Graph.h"

#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>

class Application
{
public:
    Application();
    virtual ~Application();
    void Run();

private:
    void ReadGraph(const string& file_name);

    Graph graph_;
};


#endif	/* APPLICATION_H */
