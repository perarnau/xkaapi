#include "PajeGenerator.hpp"
#include <stdio.h>
#include <math.h>

PajeGenerator::PajeGenerator  (){
    _dest = NULL;
}

 
PajeGenerator::~PajeGenerator (){
}

int PajeGenerator::initTrace (
  const std::string& name, 
  int depth, 
  int procNbr, 
  int stateTypeNbr, 
  int eventTypeNbr, 
  int linkTypeNbr, 
  int varNbr)
{
    int i;
    _dest = fopen (name.c_str(), "w");
    if (_dest ==0) return -1;

    /* Writing Header */
    fprintf (_dest, "%%EventDef PajeDefineContainerType 1\n");
    fprintf (_dest, "%% Alias string \n");
    fprintf (_dest, "%% ContainerType string \n");
    fprintf (_dest, "%% Name string \n");
    fprintf (_dest, "%%EndEventDef \n");
    fprintf (_dest, "%%EventDef PajeDefineStateType 3\n");
    fprintf (_dest, "%% Alias string \n");
    fprintf (_dest, "%% ContainerType string \n");
    fprintf (_dest, "%% Name string \n");
    fprintf (_dest, "%%EndEventDef \n");
    fprintf (_dest, "%%EventDef PajeDefineEventType 4\n");
    fprintf (_dest, "%% Alias 	string\n");
    fprintf (_dest, "%% ContainerType string\n");
    fprintf (_dest, "%% Name 		string\n");
    fprintf (_dest, "%%EndEventDef \n");
    fprintf (_dest, "%%EventDef PajeDefineEntityValue 6\n");
    fprintf (_dest, "%% Alias string  \n");
    fprintf (_dest, "%% EntityType string  \n");
    fprintf (_dest, "%% Name string  \n");
    fprintf (_dest, "%% Color color \n");
    fprintf (_dest, "%%EndEventDef  \n");
    fprintf (_dest, "%%EventDef PajeCreateContainer 7\n");
    fprintf (_dest, "%% Time date  \n");
    fprintf (_dest, "%% Alias string  \n");
    fprintf (_dest, "%% Type string  \n");
    fprintf (_dest, "%% Container string \n");
    fprintf (_dest, "%% Name string  \n");
    fprintf (_dest, "%%EndEventDef  \n");
    fprintf (_dest, "%%EventDef PajeDestroyContainer 8\n");
    fprintf (_dest, "%% Time date  \n");
    fprintf (_dest, "%% Name string  \n");
    fprintf (_dest, "%% Type string  \n");
    fprintf (_dest, "%%EndEventDef  \n");
    fprintf (_dest, "%%EventDef PajeSetState 10\n");
    fprintf (_dest, "%% Time date  \n");
    fprintf (_dest, "%% Type string  \n");
    fprintf (_dest, "%% Container string  \n");
    fprintf (_dest, "%% Value string  \n");
    fprintf (_dest, "%%EndEventDef \n");
    fprintf (_dest, "%%EventDef PajeNewEvent 20\n");
    fprintf (_dest, "%% Time          date\n");
    fprintf (_dest, "%% Type 	      string\n");
    fprintf (_dest, "%% Container     string\n");
    fprintf (_dest, "%% Value         string\n");
    fprintf (_dest, "%%EndEventDef\n");
    fprintf (_dest, "%%EventDef PajeDefineLinkType 41\n");
    fprintf (_dest, "%% Alias string\n");
    fprintf (_dest, "%% Name string\n");
    fprintf (_dest, "%% ContainerType string\n");
    fprintf (_dest, "%% SourceContainerType string\n");
    fprintf (_dest, "%% DestContainerType string\n");
    fprintf (_dest, "%%EndEventDef\n");
    fprintf (_dest, "%%EventDef PajeStartLink 42\n");
    fprintf (_dest, "%% Time date\n");
    fprintf (_dest, "%% Type string\n");
    fprintf (_dest, "%% Container string\n");
    fprintf (_dest, "%% SourceContainer string\n");
    fprintf (_dest, "%% Value string\n");
    fprintf (_dest, "%% Key string\n");
    fprintf (_dest, "%%EndEventDef\n");
    fprintf (_dest, "%%EventDef PajeEndLink 43\n");
    fprintf (_dest, "%% Time date\n");
    fprintf (_dest, "%% Type string\n");
    fprintf (_dest, "%% Container string\n");
    fprintf (_dest, "%% DestContainer string\n");
    fprintf (_dest, "%% Value string\n");
    fprintf (_dest, "%% Key string\n");
    fprintf (_dest, "%%EndEventDef\n"); 
    fprintf (_dest, "%%EventDef PajeDefineVariableType 50\n");
    fprintf (_dest, "%% Alias string\n");
    fprintf (_dest, "%% Name  string\n");
    fprintf (_dest, "%% ContainerType string \n");
    fprintf (_dest, "%%EndEventDef \n");
    fprintf (_dest, "%%EventDef PajeSetVariable 51\n");
    fprintf (_dest, "%% Time date \n");
    fprintf (_dest, "%% Type string \n");
    fprintf (_dest, "%% Container string \n");
    fprintf (_dest, "%% Value double \n");
    fprintf (_dest, "%%EndEventDef  \n");
    fprintf (_dest, "%%EventDef PajeAddVariable 52\n");
    fprintf (_dest, "%% Time date \n");
    fprintf (_dest, "%% Type string \n");
    fprintf (_dest, "%% Container string \n");
    fprintf (_dest, "%% Value double \n");
    fprintf (_dest, "%%EndEventDef  \n");
    fprintf (_dest, "%%EventDef PajeSubVariable 53\n");
    fprintf (_dest, "%% Time date \n");
    fprintf (_dest, "%% Type string \n");
    fprintf (_dest, "%% Container string \n");
    fprintf (_dest, "%% Value double \n");
    fprintf (_dest, "%%EndEventDef\n");

    fflush (_dest);

    fprintf (_dest, "1 C_Leaf0 0 'Prog'\n");
    // Defining tree
    for (i=1 ; i<depth ; i++)
        fprintf (_dest, "1 C_Leaf%d C_Leaf%d 'ProcLvl%d'\n", i, i-1, i);

    // States associated
    fprintf (_dest, "3 ST_ThreadState C_Leaf0 'Thread State'\n");


    for (i=0 ; i<eventTypeNbr ; i++)
        fprintf (_dest, "4 Event_%d C_Leaf%d 'Event_%d'\n", i, (i+3)%(depth-1), i);
    for (i=0 ; i<stateTypeNbr ; i++)
        fprintf (_dest, "6 State_%d ST_ThreadState 'State_%d' '%f %f %f'\n", i, i, (float)i/stateTypeNbr, 0.5, 1-(float)i/stateTypeNbr);

    for (i=0 ; i<linkTypeNbr ; i++)
        fprintf (_dest, "41 L_%d link%d C_Leaf0 C_Leaf%d C_Leaf%d\n", i, i%(depth-1), i%(depth-1), (i+3)%(depth-1));

    fprintf (_dest, "50 V_Memoire CPT C_Leaf%d\n", depth-1);

    /* Create variable for root container */
    fprintf (_dest, "7 0.000000 C_Proc0 C_Leaf0 0 'ROOT'\n");
    fflush (_dest);

    for (i=0 ; i<procNbr ; ++i)
    {
        fprintf (_dest, "7 0.000000 C_Thread%d C_Leaf0 C_Proc0 'Thread%d'\n", i, i);
    }
#if 0
    // Build binary tree for containers
    k = 1;
    int l;
    int m;
    for (i=1 ; i<depth-1 ; i++){
        l = (1<<(i-1))-1;
        m=0;
        if (l<0)
            l=0;
        for (j=0 ; j<(1<<i) ;){
            fprintf (_dest, "7 0.000000 C_Proc%d C_Leaf%d C_Proc%d 'C_Proc%d'\n", k, i, l, k);
            k++;
            j++;
            m++;
            if (2&m){
                l++;
                m=0;
            }
        }
    }
    fflush (_dest);
    k = (1<<depth-2)-1; // current parent proc
    j = ceil (procNbr/(1<<depth-2)); // Number of thread per proc
    // Build last proc line
    for (i=0 ; i<procNbr ;){
        fprintf (_dest, "7 0.000000 C_Thread%d C_Leaf%d C_Proc%d 'C_Leaf%d'\n", i, depth-1, k, i);
        i++;
        if (i%j == 0)
            k++;
    }
#endif



    for (i=0 ; i<varNbr ; i++){
        fprintf (_dest, "51 0.000001 V_Memoire C_Thread%d %d.000000\n", i%(procNbr), i);
    }

    fflush (_dest);
    return 0;
}

void PajeGenerator::addState  (int proc, int state, double time){
    fprintf (_dest, "10 %lf ST_ThreadState C_Thread%d State_%d\n", time, proc, state);
    fflush  (_dest);
}

void PajeGenerator::startLink (int proc, int type , double time){
    fprintf (_dest, "42 %lf L_%d C_Leaf0 C_Thread%d Value_%d Key_%d\n", time, type, proc, type, type);
    fflush  (_dest);
}

void PajeGenerator::endLink   (int proc, int type , double time){
    fprintf (_dest, "43 %lf L_%d C_Leaf0 C_Thread%d Value_%d Key_%d\n", time, type, proc, type, type);
    fflush  (_dest);
}

void PajeGenerator::addEvent  (int proc, int type , double time){
    fprintf (_dest, "20 %lf Event_%d C_Thread%d 'Event%d'\n", time, type, proc, type);
}

void PajeGenerator::incCpt    (int proc, int var  , double time){
    fprintf (_dest, "51 %lf V_Memoire C_Thread%d %lf\n", time, proc, var+time);
}

void PajeGenerator::decCpt    (int proc, int var  , double time){
    fprintf (_dest, "51 %lf V_Memoire C_Thread%d %lf\n", time, proc, ((double)var)/time);
}

void PajeGenerator::endTrace  (){
    // Destruction of containers

    fclose (_dest);
}
