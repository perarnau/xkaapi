/**************************************************************************
 *
 * N-Queens Solutions ver3.1 takaken July/2003 
 * C++ification and optimization by Serge Guelton
 * 
 **************************************************************************/

#include <iostream>
#include <athapascan-1>

/*
 * globals
 */
#define THRESHOLD 5

#define MAXSIZE 24

int SIZEE; /* SIZE -1 */
Util::ka_int32_t TOPBIT;
Util::ka_int32_t MASK;
Util::ka_int32_t SIDEMASK;

/*
 * store various counter
 */
struct res_t
{
  Util::ka_uint64_t COUNT2, COUNT4, COUNT8;

  friend std::ostream& operator<< (std::ostream & out, const res_t & r);

  res_t ():COUNT2(0),COUNT4(0), COUNT8(0)  {}

  res_t & operator+= (const res_t & r)
  {
    COUNT2 += r.COUNT2;
    COUNT4 += r.COUNT4;
    COUNT8 += r.COUNT8;
    return *this;
  }
};

/*
 * communicate res_t content
 */
a1::OStream & operator<< (a1::OStream & s_out, const res_t & b)
{
  return s_out << b.COUNT2 << b.COUNT4 << b.COUNT8;
}

a1::IStream & operator>> (a1::IStream & s_in, res_t & b)
{
  return s_in >> b.COUNT2 >> b.COUNT4 >> b.COUNT8;
}

/*
 * various small functor
 */


struct cumulinplace
{
  void operator()  (a1::Shared_rw<res_t> left,a1::Shared_r<res_t> right)
  {
    left.access() += right.read();
  }
};

struct copy
{
  void operator()  (a1::Shared_w<res_t> w,a1::Shared_r<res_t> r)
  {
    w.write(r.read());
  }
};

/*
 * display result and time
 */
struct display
{
  void operator () (a1::Shared_r < res_t > a1TOTAL,double start)
  {
    const res_t& TOTAL = a1TOTAL.read();
    double stop = Util::WallTimer::gettime(); 
    Util::logfile() << "**************************************" << std::endl;
    Util::logfile() << "n         : " << SIZEE+1 << std::endl;
    Util::logfile() << "unique    : " << (TOTAL.COUNT2+TOTAL.COUNT4+TOTAL.COUNT8) << std::endl;
    Util::logfile() << "total     : " << (TOTAL.COUNT2*2+TOTAL.COUNT4*4+TOTAL.COUNT8*8) << std::endl;
    Util::logfile() << "time      : " << stop-start << " sec" << std::endl;
    Util::logfile() << "**************************************" << std::endl;
  
  }
};


/*
 * store chess board
 */

struct board
{
  board () { }

  board (const board & b)
  {
    for(int i=0; i<SIZEE;++i){ d[i]=b.d[i]; }
  }
  Util::ka_int32_t &operator[] (int i)
  {
    return d[i];
  }

  const Util::ka_int32_t &operator[] (int i) const
  {
    return d[i];
  }

  Util::ka_int32_t d[MAXSIZE];

};

/*
 * communicate board
 */
a1::OStream & operator<< (a1::OStream & s_out, const board & b)
{
  s_out.write ( Util::WrapperFormat<Util::ka_uint32_t>::format, a1::OStream::DA, &b, SIZEE+1);
  return s_out;
}

a1::IStream & operator>> (a1::IStream & s_in, board & b)
{
  s_in.read (Util::WrapperFormat<Util::ka_uint32_t>::format, a1::OStream::DA, &b, SIZEE+1);
  return s_in;
}

/***************************************************************
 *
 *        main sequential functions
 *
 ***************************************************************/

struct param_bt1
{
  param_bt1(Util::ka_int32_t b1, board& b)
    :BOUND1(b1),BOARD(b),TOTAL()
  {
  }

  void backtrack1 (Util::ka_int32_t y, Util::ka_int32_t left, Util::ka_int32_t down, Util::ka_int32_t right)
  {
    Util::ka_int32_t bitmap = MASK & ~(left | down | right);
    if (y == SIZEE)
    {
      if (bitmap)
      {
        BOARD[y] = bitmap;
        ++TOTAL.COUNT8;
      }
    }
    else
    {
      if (y < BOUND1)
      {
        bitmap |= 2;
        bitmap ^= 2;
      }
      while (bitmap)
      {
        Util::ka_int32_t bit=-bitmap & bitmap;
        bitmap ^= BOARD[y] = bit;
        backtrack1 (y + 1, (left | bit) << 1, down | bit, (right | bit) >> 1);
      }
    }
  }

  Util::ka_int32_t BOUND1;
  board& BOARD;
  res_t TOTAL;
};


struct param_bt2
{
  param_bt2(board& b, Util::ka_int32_t lm, Util::ka_int32_t b1, Util::ka_int32_t eb)
    :BOARD(b),LASTMASK(lm),BOUND1(b1),ENDBIT(eb), BOUND2(SIZEE-b1)
  {
  }

  void check ()
  {
    /*
     * 90-degree rotation 
     */
    if (BOARD[BOUND2] == 1)
    {
      Util::ka_int32_t own=1;
      const Util::ka_int32_t *own_val=BOARD.d+1;
      for (Util::ka_int32_t ptn = 2; own <= SIZEE; own++, ptn <<= 1,own_val++)
      {
        Util::ka_int32_t bit = 1;
        for (const Util::ka_int32_t *you = BOARD.d+SIZEE; *you != ptn && *own_val >= bit; you--)
          bit <<= 1;
        if (*own_val > bit)
          return;
        if (*own_val < bit)
          break;
      }
      if (own > SIZEE)
      {
        ++TOTAL.COUNT2;
        return;
      }
    }
    /*
     * 180-degree rotation 
     */
    if (BOARD[SIZEE] == ENDBIT)
    {
      Util::ka_int32_t own=1;
      const Util::ka_int32_t *own_val= 1+BOARD.d;
      for (const Util::ka_int32_t* you = BOARD.d+SIZEE - 1; own <= SIZEE; own++,own_val++, you--)
      {
        Util::ka_int32_t bit = 1;
        for (Util::ka_int32_t ptn = TOPBIT; ptn != *you && *own_val >= bit; ptn >>= 1)
          bit <<= 1;
        if (*own_val > bit)
          return;
        if (*own_val < bit)
          break;
      }
      if (own > SIZEE)
      {
        ++TOTAL.COUNT4;
        return;
      }
    }
    /*
     * 270-degree rotation 
     */

    if (BOARD[BOUND1] == TOPBIT)
    {
      Util::ka_int32_t own=1;
      const Util::ka_int32_t *own_val= 1+BOARD.d;
      for (Util::ka_int32_t ptn = TOPBIT >> 1; own <= SIZEE; own++,own_val++, ptn >>= 1)
      {
        Util::ka_int32_t bit = 1;
        for (const Util::ka_int32_t *you = BOARD.d; *you != ptn && *own_val >= bit; you++)
          bit <<= 1;
        if (*own_val > bit)
          return;
        if (*own_val < bit)
          break;
      }
    }
    ++TOTAL.COUNT8;
  }

    void backtrack2 (Util::ka_int32_t y, Util::ka_int32_t left, Util::ka_int32_t down, Util::ka_int32_t right)
    {
      Util::ka_int32_t bitmap = MASK & ~(left | down | right);
      if (y == SIZEE) {
        if ( bitmap && !(bitmap & LASTMASK) )
        {
          BOARD[SIZEE] = bitmap;
          check ();
        }
        return;
      }
      
      if (y < BOUND1)
      {
        bitmap |= SIDEMASK;
        bitmap ^= SIDEMASK;
      }
      else if (y == BOUND2)
      {
        if (!(down & SIDEMASK))
          return;
        if ((down & SIDEMASK) != SIDEMASK)
          bitmap &= SIDEMASK;
      }
      while (bitmap)
      {
        Util::ka_int32_t bit = -bitmap & bitmap;
        bitmap ^= BOARD[y] = bit;
        backtrack2( y+1, (left | bit) << 1, down | bit,
            (right | bit) >> 1);
      }
    }

  board& BOARD;
  res_t TOTAL;

  Util::ka_int32_t LASTMASK;
  Util::ka_int32_t BOUND1;
  Util::ka_int32_t ENDBIT;

  Util::ka_int32_t BOUND2;

};



/***********************************************************************
 *
 *      athapascanized functions
 *
 ***********************************************************************/

struct bt2
{
  void operator()  (Util::ka_int32_t y, Util::ka_int32_t left, Util::ka_int32_t down, Util::ka_int32_t right, Util::ka_int32_t LASTMASK, Util::ka_int32_t BOUND1, Util::ka_int32_t ENDBIT, board BOARD, a1::Shared_w<res_t> a1TOTAL)
  {
    if ( y == THRESHOLD)
    {
      param_bt2 pbt2(BOARD, LASTMASK, BOUND1, ENDBIT);
      pbt2.backtrack2( y, left, down, right);
      a1TOTAL.write(pbt2.TOTAL);
    }
    else
    {
      Util::ka_int32_t bitmap = MASK & ~(left | down | right);

      if (y < BOUND1)
      {
        bitmap |= SIDEMASK;
        bitmap ^= SIDEMASK;
      }

      if (bitmap)
      {
        a1::Shared<res_t> a1TOTALc;
        do
        {
          Util::ka_int32_t bit=-bitmap & bitmap;
          bitmap ^= BOARD[y] = bit;
          a1::Shared<res_t> a1TOTALt;
          a1::Fork< bt2 >() (1+y, (left | bit) << 1, down | bit,(right | bit) >> 1,LASTMASK, BOUND1,ENDBIT,  BOARD, a1TOTALt);
          a1::Fork<cumulinplace>(a1::SetLocal)(a1TOTALc,a1TOTALt);
        } while(bitmap);
        a1::Fork< ::copy >()(a1TOTAL,a1TOTALc);
      }
    }
  }
};

struct bt1 {

  void operator()(Util::ka_int32_t y, Util::ka_int32_t left, Util::ka_int32_t down, Util::ka_int32_t right, Util::ka_int32_t BOUND1, board BOARD, a1::Shared_w<res_t> a1TOTAL)
  {
    if (y == THRESHOLD) 
    {
      param_bt1 pbt1(BOUND1,BOARD);
      pbt1.backtrack1 ( y,left,down,right) ;
      a1TOTAL.write(pbt1.TOTAL) ;
    }
    else
    {
      Util::ka_int32_t bitmap = MASK & ~(left | down | right);
      if (y < BOUND1)
      {
        bitmap |= 2;
        bitmap ^= 2;
      }
      if (bitmap)
      {
        a1::Shared<res_t> a1TOTALc;
        do
        {
          Util::ka_int32_t bit=-bitmap & bitmap;
          bitmap ^= BOARD[y] = bit;

          a1::Shared<res_t> a1TOTALt;
          a1::Fork< bt1 >() (y + 1, (left | bit) << 1, down | bit,(right | bit) >> 1, BOUND1, BOARD, a1TOTALt);
          a1::Fork<cumulinplace>(a1::SetLocal)(a1TOTALc,a1TOTALt);
        } while(bitmap);
        a1::Fork< ::copy >()(a1TOTAL,a1TOTALc);

      }
    }
  }
};


/**********************************************/
/*
 * Search of N-Queens 
 */
/**********************************************/

#define COMMAND(NAME)  NAME

struct COMMAND(NQueens)
{
  void operator () (Util::ka_int32_t , char **)
  {
    /*
     * Initialize 
     */
    a1::Shared < res_t > a1TOTAL ;
    board BOARD;
    double start =  Util::WallTimer::gettime();

    /*
     * 0:000000001 
     */
    /*
     * 1:011111100 
     */
    BOARD[0] = 1;
    for (Util::ka_int32_t BOUND1 = 2; BOUND1 < SIZEE; BOUND1++)
    {
      Util::ka_int32_t bit = 1 << BOUND1;
      BOARD[1] = bit;
      a1::Shared<res_t> TOTAL;
      a1::Fork< bt1 >()(2, (2 | bit) << 1, 1 | bit, bit >> 1, BOUND1, BOARD,  TOTAL);
      a1::Fork< cumulinplace >(a1::SetLocal) (a1TOTAL,TOTAL); 
    }

    /*
     * 0:000001110 
     */
    Util::ka_int32_t LASTMASK = TOPBIT | 1;
    Util::ka_int32_t ENDBIT = TOPBIT >> 1;
    Util::ka_int32_t BOUND1;
    for (BOUND1 =1 ; 2 * BOUND1 < SIZEE; BOUND1++)
    {
      Util::ka_int32_t bit = 1 << BOUND1;
      BOARD[0] = bit;
      a1::Shared<res_t> TOTAL;
      a1::Fork < bt2 > ()(1, bit << 1, bit, bit >> 1, LASTMASK, BOUND1, ENDBIT, BOARD,TOTAL);
      a1::Fork< cumulinplace >(a1::SetLocal) (a1TOTAL,TOTAL);
      LASTMASK |= LASTMASK >> 1 | LASTMASK << 1;
      ENDBIT >>= 1;
    }

    a1::Fork < display > ()(a1TOTAL,start);
  }
};


/**********************************************/
/*
 * N-Queens Solutions MAIN 
 */
/**********************************************/
#if defined(KAAPI_USE_IPHONEOS)
void* KaapiMainThread::run_main(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
  // Parse kaapi args and initialize community
  a1::Community com = a1::System::initialize_community( argc, argv);
 
  // Parse application args
  int niter = 1;
  int n= 12;
  if (argc >=2) {
    n = atoi(argv[1]);
  }
  if (argc >=3) {
    niter = atoi(argv[2]);
  }
  SIZEE = n-1;
  TOPBIT = (1 << SIZEE);
  MASK = ((1 << (SIZEE+1)) - 1);
  SIDEMASK = ( ( 1 << SIZEE) | 1);

  // Commit community, ie start kaapi runtime
  com.commit();
//Util::KaapiComponentManager::prop deprecated  Util::logfile() << "[main] pid = " << getpid() << " nb_local_threads = " << Util::KaapiComponentManager::prop["community.thread.poolsize"] << std::endl;

  for (int i=0; i<niter; ++i)
  {
    a1::ForkMain < COMMAND(NQueens) > ()(argc, argv);
    a1::Sync();
  }

  com.leave ();

  /* */
  a1::System::terminate ();
  return 0;
}
