#include "Application.h"
#include "CuFunctionsManager.h"
#include "GeneticSolver.h"

#include <cuda.h>
#include "utils.h"
#include <climits>
using namespace std;

Application::Application()
{
    srand(time(NULL));
}

void Application::Run()
{
    cout << "Type test name: ";
    string test_name;
    cin >> test_name;
    cout << "\n";

    ReadGraph(test_name + ".in");

    GeneticSolver GS;
    GS.SetGraph(&graph_);
    GS.Prepare();

    cout << "Structure created -> start of computation\nThe best output will be stored in " << test_name << ".out file\n\nActual best score:\n";

    int actual_best_score = INT_MAX;
    while(true)
    {
        REP(i, 20)
            GS.RunStage();

        int new_score = GS.GetBestScore();
        if(new_score < actual_best_score)
        {
            actual_best_score = new_score;
            cout << "__________________" << GS.GetBestScore() << "\n";

            fstream output_file(test_name + ".out", ios::out | ios::trunc);

            output_file << "Score: " << actual_best_score << "\n\nSolution:\n";

            int* best_solution = GS.GetLeader();
            REP(i, graph_.size)
                output_file << best_solution[i] << " ";
            output_file << "\n";
            output_file.close();
        }
    }
}

void Application::ReadGraph(const string& file_name)
{
    fstream file(file_name, ios::in);
    file >> graph_;
    file.close();
}

Application::~Application()
{

}


