// =========================================================================
// 
// Author: T. Gautier
// Status: ok
//
// =========================================================================
#ifndef _MY_LIST_H_FOR_LR_
#define _MY_LIST_H_FOR_LR_
#include <iostream>
#include <stddef.h> /* size_t */

/*========================================== LIST MANAGEMENT ==========================================*/

/* fwd decl */
class sublist;

/* List data structure: stored into an array 
   The last element of the list has its index to the successor equal to -n the size of the list.
*/
class list {
public:
  typedef long int index_t;

  typedef unsigned long int size_t;
  
  /* node of list */
  struct node
  {
    long int nS; // Position of the successor in the array
    long int R;  // Rank
  };
  
  /* Initialize the list such as initial R = 1 and all successors == -n
  */
  void initialize();
  
  /* Allocate the memory for a list of size n elements
  */
  void resize(size_t n);

  /* This version generates random ranks, and after that it links each node with its successor.
     The last element of the list has its successor value == -n.
     Return the head of the list.
  */
  long int randomize();

  /* print a list of n elements 
  */
  void print( std::ostream& ) const;

  /* Verify the result of the list ranking 
  */
  int testOK() const;

  /* Compute the sum of all successor index.
     See Helman & Jaja, design pratical efficient algorithms for symmetric multiprocessors.
  */
  index_t sum_successor( ) const;

  /* Return the head index of the list.
     n*(n-1)/2 - sum_successor_list(L, n)
     See Helman & Jaja, design pratical efficient algorithms for symmetric multiprocessors.
  */
  index_t head( ) const;

  /* Do list ranking on the sublist starting at position h_head. 
     nh is the successor of the head.
     On return lR contains the last rank computed for the sublist.
     The return value is the index of the last element of the (sub) list.
     The next element of the return value is either the marker of the end of the list,
     or the head of the next sublist.
  */
  index_t lr_head( index_t h_head, index_t nh, index_t& lR);

  /* Split the list L[0], ..., L[n-1] into s sublist and add
     description of splitters into sL[s_beg], .., sL[s_end-1].
     For the j-th splitter with index k into L, then on return we have
        * sL[j].save_nS = L[k].nS and L[k].nS = -(j+1)
     Return the number of splitters: If after 100 random selections, a new splitter cannot
     be selected, then it was not generated.
  */
  index_t  split( sublist* sL, long int s_beg, long int s_end );
  
  /* access to the element ith of the underlaying array */
  const node& operator[](index_t i) const
  { return _rep[i]; }

  /* access to the element ith of the underlaying array */
  node& operator[](index_t i) 
  { return _rep[i]; }

protected:
  index_t _size;
  node*   _rep;
};



class sublist {
public:
  typedef list::index_t index_t;
  
public:
  long int head;     // Position of the head of the sublist into the list L
  long int next;     // Index of the next sublist
  long int save_nS;  // copy of the successor of L[j].ns 
  long int R;        // The last rank of the sublist
  long int pR;       // Prefix of the rank to add for each sublist
};


#endif