/**************************************************************************
 *
 * N-Queens Solutions ver3.1 takaken July/2003 
 * C++ification and optimization by S. Guelton, T. Gautier
 * 
 **************************************************************************/

#include <iostream>
#include <kaapi++>

/*
 * globals
 */
#if !defined(THRESHOLD)
#define THRESHOLD 8
#endif

#define MAXSIZE 24

int SOL;   /* SOL for debug */
#if !defined(SIZE)
int SIZEE;                /* SIZE -1 */
int32_t TOPBIT;    /* 1 << SIZEE */
int32_t MASK;      /*  MASK = ((1 << (SIZEE+1)) - 1); */
int32_t SIDEMASK;  /* SIDEMASK = ( ( 1 << SIZEE) | 1); */
#else
#define SIZEE     (SIZE-1)
#define TOPBIT    (1 << SIZEE)
#define MASK      ((1 << (SIZEE+1)) - 1)
#define SIDEMASK  ((1 << SIZEE) | 1)
#endif

/*
 * store various counter
 */
struct res_t
{
  uint64_t COUNT2, COUNT4, COUNT8;

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

#if 0
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
#endif

/*
 * various small functor
 */


struct Task_cumulinplace: ka::Task<2>::Signature<
  ka::RW<res_t>,
  ka::R<res_t>
> {};
template<>
struct TaskBodyCPU<Task_cumulinplace> {
  void operator()  (ka::pointer_rw<res_t> left, ka::pointer_r<res_t> right)
  {
    *left += *right;
  }
};


struct Task_copy: ka::Task<2>::Signature<
  ka::W<res_t>,
  ka::R<res_t>
> {};

template<>
struct TaskBodyCPU<Task_copy> {
  void operator()  (ka::pointer_w<res_t> w, ka::pointer_r<res_t> r)
  {
    *w = *r;
  }
};

/*
 * display result and time
 */
struct Task_display: ka::Task<2>::Signature<
  ka::R<res_t>,
  double
> {};
template<>
struct TaskBodyCPU<Task_display> {
  void operator () (ka::pointer_r<res_t> TOTAL, double start)
  {
    double stop = ka::WallTimer::gettime(); 
    std::cout << "**************************************" << std::endl;
    std::cout << "n         : " << SIZEE+1 << std::endl;
    std::cout << "unique    : " << (TOTAL->COUNT2 + TOTAL->COUNT4 + TOTAL->COUNT8) << std::endl;
    std::cout << "total     : " << (TOTAL->COUNT2*2 + TOTAL->COUNT4*4 + TOTAL->COUNT8*8) << std::endl;
    std::cout << "time      : " << stop-start << " sec" << std::endl;
    std::cout << "**************************************" << std::endl;
    if ((SOL != -1) && ((unsigned)SOL != (TOTAL->COUNT2*2 + TOTAL->COUNT4*4 + TOTAL->COUNT8*8))) abort(); 
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
  int32_t &operator[] (int i)
  {
    return d[i];
  }

  const int32_t &operator[] (int i) const
  {
    return d[i];
  }

  int32_t d[MAXSIZE];
};

/*
 * communicate board
 */
#if 0 // TODO 
a1::OStream & operator<< (a1::OStream & s_out, const board & b)
{
  s_out.write ( ka::WrapperFormat<uint32_t>::format, a1::OStream::DA, &b, SIZEE+1);
  return s_out;
}

a1::IStream & operator>> (a1::IStream & s_in, board & b)
{
  s_in.read (Util::WrapperFormat<uint32_t>::format, a1::OStream::DA, &b, SIZEE+1);
  return s_in;
}
#endif

/***************************************************************
 *
 *        main sequential functions
 *
 ***************************************************************/
struct param_bt1
{
  param_bt1(int32_t b1, board& b)
    :BOUND1(b1),BOARD(b),TOTAL()
  {
  }

  void backtrack1 (int32_t y, int32_t left, int32_t down, int32_t right)
  {
    int32_t bitmap = MASK & ~(left | down | right);
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
        int32_t bit=-bitmap & bitmap;
        bitmap ^= BOARD[y] = bit;
        backtrack1 (y + 1, (left | bit) << 1, down | bit, (right | bit) >> 1);
      }
    }
  }

  int32_t BOUND1;
  board& BOARD;
  res_t TOTAL;
};


struct param_bt2
{
  param_bt2(board& b, int32_t lm, int32_t b1, int32_t eb)
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
      int32_t own=1;
      const int32_t *own_val=BOARD.d+1;
      for (int32_t ptn = 2; own <= SIZEE; own++, ptn <<= 1,own_val++)
      {
        int32_t bit = 1;
        for (const int32_t *you = BOARD.d+SIZEE; *you != ptn && *own_val >= bit; you--)
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
      int32_t own=1;
      const int32_t *own_val= 1+BOARD.d;
      for (const int32_t* you = BOARD.d+SIZEE - 1; own <= SIZEE; own++,own_val++, you--)
      {
        int32_t bit = 1;
        for (int32_t ptn = TOPBIT; ptn != *you && *own_val >= bit; ptn >>= 1)
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
      int32_t own=1;
      const int32_t *own_val= 1+BOARD.d;
      for (int32_t ptn = TOPBIT >> 1; own <= SIZEE; own++,own_val++, ptn >>= 1)
      {
        int32_t bit = 1;
        for (const int32_t *you = BOARD.d; *you != ptn && *own_val >= bit; you++)
          bit <<= 1;
        if (*own_val > bit)
          return;
        if (*own_val < bit)
          break;
      }
    }
    ++TOTAL.COUNT8;
  }

    void backtrack2 (int32_t y, int32_t left, int32_t down, int32_t right)
    {
      int32_t bitmap = MASK & ~(left | down | right);
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
        int32_t bit = -bitmap & bitmap;
        bitmap ^= BOARD[y] = bit;
        backtrack2( y+1, (left | bit) << 1, down | bit,
            (right | bit) >> 1);
      }
    }

  board& BOARD;
  res_t TOTAL;

  int32_t LASTMASK;
  int32_t BOUND1;
  int32_t ENDBIT;
  int32_t BOUND2;
};



/***********************************************************************
 *
 *      kaapi tasks
 *
 ***********************************************************************/

struct Task_bt2: ka::Task<9>::Signature<
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  board,
  ka::W<res_t>
> {};

template<>
struct TaskBodyCPU<Task_bt2>  {
  void operator()(
      ka::Thread* thread,
      int32_t y, 
      int32_t left, 
      int32_t down, 
      int32_t right, 
      int32_t LASTMASK, 
      int32_t BOUND1, 
      int32_t ENDBIT, 
      board BOARD, 
      ka::pointer_w<res_t> a1TOTAL
  )
  {
    if ( y == THRESHOLD)
    {
      param_bt2 pbt2(BOARD, LASTMASK, BOUND1, ENDBIT);
      pbt2.backtrack2( y, left, down, right);
      *a1TOTAL = pbt2.TOTAL;
    }
    else
    {
      int32_t bitmap = MASK & ~(left | down | right);

      if (y < BOUND1)
      {
        bitmap |= SIDEMASK;
        bitmap ^= SIDEMASK;
      }

      if (bitmap)
      {
        ka::pointer<res_t> a1TOTALc = thread->Alloca<res_t>();
        do
        {
          int32_t bit=-bitmap & bitmap;
          bitmap ^= BOARD[y] = bit;
          ka::pointer<res_t> a1TOTALt = thread->Alloca<res_t>();
          thread->Spawn<Task_bt2>() (1+y, (left | bit) << 1, down | bit,(right | bit) >> 1,LASTMASK, BOUND1,ENDBIT,BOARD,a1TOTALt);
          thread->Spawn<Task_cumulinplace>()(a1TOTALc,a1TOTALt);
        } while(bitmap);
        thread->Spawn<Task_copy >()(a1TOTAL,a1TOTALc);
      }
    }
  }
};


struct Task_bt1: ka::Task<7>::Signature<
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  int32_t,
  board,
  ka::W<res_t>
> {};


template<>
struct TaskBodyCPU<Task_bt1> {
  void operator()(
      ka::Thread* thread,
      int32_t y, 
      int32_t left, 
      int32_t down, 
      int32_t right, 
      int32_t BOUND1, 
      board BOARD, 
      ka::pointer_w<res_t> a1TOTAL
  )
  {
    if (y == THRESHOLD) 
    {
      param_bt1 pbt1(BOUND1,BOARD);
      pbt1.backtrack1 ( y,left,down,right) ;
      *a1TOTAL = pbt1.TOTAL;
    }
    else
    {
      int32_t bitmap = MASK & ~(left | down | right);
      if (y < BOUND1)
      {
        bitmap |= 2;
        bitmap ^= 2;
      }
      if (bitmap)
      {
        ka::pointer<res_t> a1TOTALc = thread->Alloca<res_t>();
        do
        {
          int32_t bit=-bitmap & bitmap;
          bitmap ^= BOARD[y] = bit;

          ka::pointer<res_t> a1TOTALt = thread->Alloca<res_t>();
          thread->Spawn<Task_bt1>() (y + 1, (left | bit) << 1, down | bit,(right | bit) >> 1, BOUND1, BOARD, a1TOTALt);
          thread->Spawn<Task_cumulinplace>()(a1TOTALc,a1TOTALt);
        } while(bitmap);
        thread->Spawn<Task_copy >()(a1TOTAL,a1TOTALc);
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
  void operator () (int32_t , char **)
  {
    /*
     * Initialize 
     */
    ka::pointer<res_t> a1TOTAL = ka::Alloca<res_t>();
    board BOARD;
    double start =  ka::WallTimer::gettime();

    /*
     * 0:000000001 
     */
    /*
     * 1:011111100 
     */
    BOARD[0] = 1;
    for (int32_t BOUND1 = 2; BOUND1 < SIZEE; BOUND1++)
    {
      int32_t bit = 1 << BOUND1;
      BOARD[1] = bit;
      ka::pointer<res_t> TOTAL = ka::Alloca<res_t>();
      ka::Spawn< Task_bt1 >()(2, (2 | bit) << 1, 1 | bit, bit >> 1, BOUND1, BOARD,  TOTAL);
      ka::Spawn< Task_cumulinplace >() (a1TOTAL,TOTAL); 
    }

    /*
     * 0:000001110 
     */
    int32_t LASTMASK = TOPBIT | 1;
    int32_t ENDBIT = TOPBIT >> 1;
    int32_t BOUND1;
    for (BOUND1 =1 ; 2 * BOUND1 < SIZEE; BOUND1++)
    {
      int32_t bit = 1 << BOUND1;
      BOARD[0] = bit;
      ka::pointer<res_t> TOTAL = ka::Alloca<res_t>();
      ka::Spawn< Task_bt2 >()(1, bit << 1, bit, bit >> 1, LASTMASK, BOUND1, ENDBIT, BOARD,TOTAL);
      ka::Spawn< Task_cumulinplace >() (a1TOTAL,TOTAL);
      LASTMASK |= LASTMASK >> 1 | LASTMASK << 1;
      ENDBIT >>= 1;
    }

    ka::Spawn<Task_display>()(a1TOTAL,start);
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
  ka::Community com = ka::System::initialize_community( argc, argv);
 
  // Parse application args
  int niter = 1;
  int n= 12;
  int sol = -1;
  if (argc >=2) {
    n = atoi(argv[1]);
  }
  if (argc >=3) {
    niter = atoi(argv[2]);
  }
  if (argc >=4) {
    sol = atoi(argv[3]);
  }
  SOL=sol;
#if !defined(SIZE)
  SIZEE = n-1;
  TOPBIT = (1 << SIZEE);
  MASK = ((1 << (SIZEE+1)) - 1);
  SIDEMASK = ( ( 1 << SIZEE) | 1);
#endif

  // Commit community, ie start kaapi runtime
  com.commit();

  for (int i=0; i<niter; ++i)
  {
    ka::SpawnMain < COMMAND(NQueens) > ()(argc, argv);
    ka::Sync();
  }

  com.leave ();

  /* */
  ka::System::terminate ();
  return 0;
}
