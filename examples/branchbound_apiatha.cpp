/**************************************************************************
 *
 * Branch & Bound examples
 * by S. Guelton
 * 
 **************************************************************************/

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include "athapascan-1"


double *v=NULL;      ///< an array of treasure volume
double *g=NULL;      ///< an array of treasure profits
int n_treas;         ///< number of different treasures 
double vMax;         ///< volume of knackpack



/** Class treasure, it contains all the operations concerning each treasure*/
struct treas{
  double volume;
  double gain;
  treas() : volume(0),gain(0){}
  treas(double v,double g) : volume(v),gain(g){}
};

/** 
  \param T1  first treasure
  \param T2  second treasure
  \return true if T1 gain/volume is bigger than T2's
*/
bool operator< (const treas & T1, const treas & T2){
  return ((T1.gain/T1.volume)>(T2.gain/T2.volume));
}
  
/** It represents a possible solution.
Each node memorizes its path of solution (treasures already in bag),
current volume of bag and current profits of bag. */

class node {
  double v_k;          ///< current volume of bag in this node
  double g_k;          ///< current profit of bag in this node
  int k;               ///< k th treasure put in bag
  int * solCurrent;    ///< path of solution for this node
  friend a1::OStream& operator<<( a1::OStream& , const node& ) ;
  friend a1::IStream& operator>>( a1::IStream&, node& ) ;

public:
/** Initialize objects*/
  node() : v_k(0), g_k(0), k(-1), solCurrent(new int[n_treas]) {
    std::fill(solCurrent,solCurrent+n_treas,0);
  }
/** Copy of this node*/
  node( const node& n):
        v_k(n.v_k), g_k(n.g_k), k(n.k), solCurrent(new int[n_treas]) {
    std::copy(n.solCurrent, n.solCurrent+n_treas, solCurrent);
  }

/** Release the memory at the end of the use of this node*/
  ~node() {
    delete []solCurrent;
  }

/** Operation '=' between node type */
  node& operator=(const node& n) {
    v_k=n.v_k;
    g_k=n.g_k;
    k=n.k;
    std::copy(n.solCurrent, n.solCurrent+n_treas, solCurrent);
    return *this;
  }

/** A possible solution*/
  bool isSolution() const {
    return true;
  }

/** An optimist solution*/
  bool evalOptimist(double gMax) const {      //bound
    return (g_k+g[k+1]*(vMax-v_k)/v[k+1])>gMax;  
  }

/** Get the current profit of this node*/
  double value() const {
    return g_k;
  }

/** Get the current solution path*/
  int * solution() const {
    return solCurrent;
  }

/** Add n another treasure into bag*/
  void addTreas(int n){
    k+=1;
    solCurrent[k]=n;
    v_k += n*v[k];
    g_k += n*g[k];  
  }

/** Remove the treasure of this node*/
  void rmTreas(){
    int n = solCurrent[k];
    v_k -= n*v[k];
    g_k -= n*g[k];
    solCurrent[k] = 0;
    k-=1;
  }

/** Insert all the children (another treasure) of this node, 
each child represents the number of the treasure that we will add */
  void insertAllChildren (std::vector<node>& nodes) const {

    if(k==n_treas-1)
      return;

    const int vol = static_cast <int> ((vMax-v_k)/v[k+1]);
    for(int i=0;i<=vol;i++) {
      node child(*this);
      child.addTreas(i);
      nodes.push_back(child);
    }
  }
};

/** max between n2 and n1 */
struct maximum{
  bool operator() (node & n1, const node & n2 ) {

    if (n2.value() > n1.value())
                {
      n1 = n2;
                        return true;
                }
                return false;
  }
};


/** Define a global valable which can be read/written by any task,
n_max is the node who has the maximus value currently*/
a1::MonotonicBound<node,maximum> n_max;


a1::OStream& operator<<( a1::OStream& out, const node& n) {
  out << n.v_k << n.g_k << n.k ;
  for(int i=0;i<n_treas;i++)
    out << n.solCurrent[i];
  return out;
}

a1::IStream& operator>>( a1::IStream& in, node& n) {
  in >> n.v_k >> n.g_k >> n.k ;
  for(int i=0;i<n_treas;i++)
    in >> n.solCurrent[i];
  return in;  
}


/** main task */
struct AthaBB {
  void operator() (a1::Shared_w <bool> syn, node n) {

    /* bound*/
    n_max.acquire();      /// take the last updated value of n_max
    const node & nMax = n_max.read();  /// read n_max's value
    if ( n.evalOptimist( nMax.value() ) ) {

      n_max.update(n);    /// update n_max
      n_max.release();

      /* branch */
      std::vector<node> childN;
      n.insertAllChildren(childN);  /// create all the children of node n
      
      /* fork all children task */
      for (size_t i=0; i < childN.size(); i++){
        /* fork child */
        a1::Fork<AthaBB>()(syn,childN[i]);
                /// do the same thing for all the children as node n in parallel
      }

    }
    else n_max.release();    
  }
};

/** Print the best solution and the number of each treasure taken for this solution*/ 
struct print {
  void operator()(a1::Shared_r<bool> r) {

    n_max.acquire();
    const node & n = n_max.read();
    std::cout<<"The best solution is:"<< n.value() <<std::endl;

    for (int i = 0; i<n_treas; i++) {
    
      std::cout <<"The number of "<< i <<
                "th treasure: "<< n.solution()[i] << std::endl;
    }
    n_max.release();
  }
};

/** Main of the program*/
struct doit {
  void operator()(int argc, char ** argv) {
    node nodebase;
    a1::Shared <bool> sync;
    a1::Fork<AthaBB>()(sync,nodebase);
    a1::Fork<print>()(sync);
  }
};


/** Default main*/
int main(int argc,char**argv){

  a1::Community com = a1::System::join_community(argc,argv);
  
  if(argc<1) {
    std::cerr << "not enough args, usage : <prog> <filename>"
        << std::endl
        << "where <filename> contains all args in the form"
        << std::endl
        << "nb_elem"
        << "gain volume"
        << std::endl
        << "..."
        << std::endl
        << "gain volume"
        << "total weight"
        << std::endl;
    return 0;
  }
  std::ifstream input(argv[1]);

  input >> n_treas ;

  v = new double[n_treas];
  g = new double[n_treas];

  treas *  tabTrea = new treas[n_treas];

  for(int i=0; i< n_treas && ! input.eof(); i++ ) {
    
    input >> tabTrea[i].gain ;
    assert(! input.eof());
    input >> tabTrea[i].volume ;
  }
  input >> vMax;

  std::sort(tabTrea,tabTrea+n_treas);/// sort treasures in order of gain/volume


  for (int i = 0; i<n_treas; i++) {

    g[i] = tabTrea[i].gain;
    v[i] = tabTrea[i].volume;
  }

  delete [] tabTrea;


  n_max.initialize("nodeMax", new node() );

  a1::ForkMain<doit>() (argc,argv);

  com.leave();
  n_max.terminate();
  a1::System::terminate();

  return 0;
}
