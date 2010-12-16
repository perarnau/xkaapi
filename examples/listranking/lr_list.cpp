#include "lr_list.h"
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>


/* =============================================================== */
list::list()
 : _size(0), _rep(0)
{
}


/* =============================================================== */
void list::resize(size_t n)
{
  if (_rep !=0) delete [] _rep;
  _rep  = new node[n];
  _size = n;
}  

/* =============================================================== */
list::index_t list::ordered()
{
	for(index_t i=0; i < _size-1; ++i)
	{
		_rep[i].nS = i+1;
		_rep[i].R = 0;
  }
  _rep[_size-1].nS = -_size;
  _rep[_size-1].R = 0;
  return 0;
}

/* =============================================================== */
/* This version generates random ranks, and after that it links each node with his successor
*/
list::index_t list::randomize()
{
#if 0
  if (_size == 1) return -1;

  /* use temporary array to retreive where is elements of rank r */
  index_t* whereisrank = new index_t[_size+1];
  for (index_t i=0; i<_size; ++i)
  {
    _rep[i].nS = i+1;
    _rep[i].R  = i;
    whereisrank[_rep[i].R] = i;
  }
  _rep[_size-1].nS = -_size;
  _rep[_size-1].R  = _size-1;
  whereisrank[_size] = -_size; 
  

  /* shuffle algorithm from Fisher-Yates, Durstenfeld and Knuth 
     http://en.wikipedia.org/wiki/Fisher-Yates_shuffle
  */
	srand(time(0));
  for (index_t i=0; i<_size-1; ++i)
  {
    index_t j = i + rand() / (RAND_MAX / (_size - i) + 1);
    whereisrank[_rep[i].R] = j;
    whereisrank[_rep[j].R] = i;
    std::swap( _rep[i], _rep[j] );
  } 

  for (index_t i=0; i<_size-1; ++i)
  {
    _rep[i].nS = whereisrank[_rep[i].R+1];
    _rep[i].R  = 0;
  }  
  index_t head = whereisrank[0];
  delete [] whereisrank;
  return head;
#else
	index_t i,pos,pos_ant;
  index_t pos_end =0;
  index_t head = 0;

  for (index_t i=0; i<_size; ++i)
  {
    _rep[i].nS = -1;
    _rep[i].R  = 0;
  }

	srand (time(NULL));
	pos = rand() % _size;
	i=0;
	pos_ant = -1;
	while (i < _size)
	{
		if (i == 0) // Last element
		{
			_rep[pos].nS = - _size;
			i++;
			pos_ant = pos;
			head = pos;
			pos_end = pos;
		}
		else
		{
			if ((_rep[pos].nS < 0) && (pos != pos_end))
			{
				_rep[pos].nS = pos_ant;
				i++;
				pos_ant = pos;
				head = pos;
			}
		}
		pos = rand() % _size;
	}
  return head;
#endif
}


/* =============================================================== */
/*
*/
void list::print( std::ostream& sout ) const
{
	for(int i=0; i < _size; i++)
	{
    sout << "L[" << i << "] ->" << "(" << _rep[i].R << "," << _rep[i].nS << ")" << std::endl;
  }
}


/* =============================================================== */
/**
*/
int list::testOK() const
{
	index_t i;
	int ok=1;
	std::vector<index_t> Vok(_size);
	std::vector<index_t> Vals(_size);
	for (i=0; i < _size; i++)
	{
		Vok[i] = 0;
		Vals[i] = ~0L;
	}
	i=0;
	while (i<_size)
	{
		if (Vok[_rep[i].R] == 1)
		{
			ok=0; // FAIL!!!
			std::cout << "Value: " << _rep[i].R << " in " << i << " already exists in " 
                << Vals[_rep[i].R] << ": " << _rep[Vals[_rep[i].R]].R 
                << std::endl;
		}
		else
		{
			Vok[_rep[i].R] = 1;
			Vals[_rep[i].R] = i;
		}
		i++;
	}
	return ok;
}


/* =============================================================== */
/*
*/
list::index_t list::sum_successor() const
{
  index_t sum = 0;
  for (index_t i=0; i<_size; ++i)
  {
    if (_rep[i].nS >0) sum += _rep[i].nS;
  }
  return sum;
}


/* =============================================================== */
/* Function sum_successor_list may be inlined into the body for better performance
*/
list::index_t list::head( ) const
{
  return (_size*(_size-1))/2 - sum_successor();
}


/* =============================================================== */
/*
*/
list::index_t list::lr_head( index_t h_head, index_t nh, index_t& lR)
{
  index_t curr = 0;
  _rep[h_head].R = curr;
  while (1)
  {
    index_t fnh = _rep[nh].nS;
    if (fnh <0) 
    {
       lR = curr; 
       return nh; 
    }
    ++curr;
    _rep[nh].R = curr;
    h_head = nh;
    nh = fnh;
  }
}



/* =============================================================== */
list::index_t list::lr_head( index_t h_head, index_t nh, index_t& lR, index_t SLindex, index_t* SLi)
{
  index_t curr = 0;
  _rep[h_head].R = curr;
  while (1)
  {
    index_t fnh = _rep[nh].nS;
    if (fnh <0) 
    {
       lR = curr; 
       return nh; 
    }
    ++curr;
    _rep[nh].R = curr;
    SLi[nh]    = SLindex;
    h_head     = nh;
    nh         = fnh;
  }
}


/* =============================================================== */
/*
*/
list::index_t list::split( sublist* sL, index_t s_beg, index_t s_end )
{
  int t;
  index_t scurr = s_beg;  
  index_t n = s_end - s_beg;
  for (index_t i=0; i< n; ++i)
  {
    index_t k;
    for (t=0; t<100; ++t)
      if (_rep[k = rand() % n].nS >=0) 
        break;
      
    if (t ==100) 
    {
      sL[scurr].head    = -1;
      sL[scurr].save_nS = -1;
      sL[scurr].next    = -1;
      std::cout << "Cannot generate header" << std::endl;
      continue; /* pass to the next generation of splitter */
    }
    
    sL[scurr].head    = k;
    sL[scurr].save_nS = _rep[k].nS;
    _rep[k].nS        = -(scurr+1);
    ++scurr;
  }
  return scurr;
}


/* =============================================================== */
/*
*/
list::index_t list::split( sublist* sL, index_t s_beg, index_t s_end, index_t SLindex_offset, index_t* SLi )
{
  int t;
  index_t scurr = s_beg;  
  index_t n = s_end - s_beg;
  for (index_t i=0; i< n; ++i)
  {
    index_t k;
    for (t=0; t<100; ++t)
      if (_rep[k = rand() % n].nS >=0) 
        break;
      
    if (t ==100) 
    {
      sL[scurr].head    = -1;
      sL[scurr].save_nS = -1;
      sL[scurr].next    = -1;
      std::cout << "Cannot generate header" << std::endl;
      continue; /* pass to the next generation of splitter */
    }
    
    sL[scurr].head    = k;
    sL[scurr].save_nS = _rep[k].nS;
    SLi[k]            = scurr+SLindex_offset;
    _rep[k].nS        = -(scurr+1);
    ++scurr;
  }
  return scurr;
}
