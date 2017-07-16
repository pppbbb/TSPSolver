#ifndef UTILS_H
#define	UTILS_H

using namespace std;

#define REP(i,x) for(int i = 0 ; i < (x) ; i++)

//#define DEBUG_MODE
#ifdef DEBUG_MODE
    //#define DEBUG_PRINT_COMMUNICATE
#endif

#ifdef DEBUG_PRINT_COMMUNICATE
    //#define DEBUG_PRINT_ARRAYS
    #define print(x) cerr << #x << " = " << x << endl
    #define debug(x) x
#else
    #define print(x)
    #define debug(x)
#endif

#endif	/* UTILS_H */