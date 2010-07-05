//
// Made by fabien le mentec <texane@gmail.com>
// 
// Started on  Wed Mar 10 20:08:49 2010 texane
// Last update Wed Mar 10 22:43:46 2010 texane
//



#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <numeric>
#include <algorithm>
#include <kaapi++>
#include <pthread.h>


#define CONFIG_DEBUG 0
//#define CONFIG_DEBUG 1


template<typename _IteratorType>
struct BasicSequence
{
  // splittable sequence

  typedef _IteratorType IteratorType;

  IteratorType _beg;
  IteratorType _end;

  BasicSequence() {}

  BasicSequence(const IteratorType& beg, const IteratorType& end)
    : _beg(beg), _end(end) {}

  BasicSequence(const BasicSequence& seq)
  {
    *this = seq;
  }

  void empty_seq(BasicSequence& seq) const
  {
    seq._beg = _beg;
    seq._end = _beg;
  }

  bool is_empty() const
  {
    return _beg == _end;
  }

  size_t size() const
  {
    return std::distance(_beg, _end);
  }

  void split(BasicSequence& seq, size_t size)
  {
    seq._beg = _beg;
    _beg += size;
    seq._end = _beg;
  }

  void rsplit(BasicSequence& seq, size_t size)
  {
    seq._end = _end;
    seq._beg = _end - size;
    _end = seq._beg;
  }

  void join(const BasicSequence& seq)
  {
    _end = seq._end;
  }
};



template<typename _InputIterator, typename _OutputIterator>
struct InOutSequence
{
  typedef _InputIterator InputIterator;
  typedef _OutputIterator OutputIterator;

  typedef InOutSequence<InputIterator, OutputIterator>
  SequenceType;

  BasicSequence<InputIterator> _iseq;
  OutputIterator _obeg;

  InOutSequence() {}

  InOutSequence
  (
   const InputIterator& ibeg,
   const InputIterator& iend,
   const OutputIterator& obeg
  )
  {
    _iseq = BasicSequence<InputIterator>(ibeg, iend);
    _obeg = obeg;
  }

  InOutSequence(const SequenceType& seq)
  {
    *this = seq;
  }

  void empty_seq(SequenceType& seq) const
  {
    _iseq.empty_seq(seq._iseq);
  }

  bool is_empty() const
  {
    return _iseq.is_empty();
  }

  size_t size() const
  {
    return _iseq.size();
  }

  void split(SequenceType& seq, size_t size)
  {
    _iseq.split(seq._iseq, size);
    seq._obeg = _obeg;
    _obeg += size;
  }

  void rsplit(SequenceType& seq, size_t size)
  {
    _iseq.rsplit(seq._iseq, size);
    seq._obeg = _obeg + std::distance(_iseq._beg, seq._iseq._beg);
  }

  void join(const SequenceType& seq)
  {
    _iseq.join(seq._iseq);
    _obeg = seq._obeg;
  }

};


template<typename _SequenceType>
struct BaseWork
{
  typedef _SequenceType SequenceType;

  SequenceType _remaining_seq;
  SequenceType _processed_seq;

#if CONFIG_DEBUG
  SequenceType _original_seq;
  bool _is_master;
#endif

  BaseWork(const SequenceType& seq)
  {
    _remaining_seq = seq;
    _remaining_seq.empty_seq(_processed_seq);

#if CONFIG_DEBUG
    _original_seq = seq;
    _is_master = true;
#endif
  }

  void prepare(const BaseWork&)
  {}

  void process(SequenceType&)
  {}

  void reduce(SequenceType&)
  {}
};


template<typename SequenceType>
struct DummyWork : BaseWork<SequenceType>
{
  DummyWork(const SequenceType& seq)
    : BaseWork<SequenceType>(seq)
  {}

  void process(SequenceType& seq)
  {
    for (unsigned int j = 0; j < 100; ++j)
      usleep(1000);

    seq._beg = seq._end;
  }

  void reduce(SequenceType& seq)
  {
    for (size_t i = seq._beg; i < seq._end; ++i)
      usleep(10000);

    seq._beg = seq._end;
  }
};


template<typename SequenceType>
struct PrefixWork : BaseWork<SequenceType>
{
  typedef typename SequenceType::InputIterator InputIteratorType;
  typedef typename SequenceType::OutputIterator OutputIteratorType;

  typedef typename std::iterator_traits
  <OutputIteratorType>::value_type ValueType;

  bool _has_value;
  unsigned int _value;

  PrefixWork(const SequenceType& seq)
    : BaseWork<SequenceType>(seq), _has_value(false)
  {}

  void prepare(const PrefixWork& work)
  {
#if CONFIG_DEBUG
    this->_original_seq = work._original_seq;
    this->_is_master = false;
#endif

    _has_value = false;
  }

  void process(SequenceType& seq)
  {
#if CONFIG_DEBUG
    printf("%c [%x] process(%lu - %lu, %lu), %lu\n",
	   this->_is_master ? 'm' : 's',
	   (unsigned int)pthread_self(),
	   std::distance(this->_original_seq._iseq._beg, seq._iseq._beg),
	   std::distance(this->_original_seq._iseq._beg, seq._iseq._end),
	   std::distance(this->_original_seq._obeg, seq._obeg),
	   seq._iseq.size());
#endif

    InputIteratorType ipos = seq._iseq._beg;
    InputIteratorType iend = seq._iseq._end;

    OutputIteratorType opos = seq._obeg;

    ValueType value;

    if (_has_value == false)
    {
      _has_value = true;
      _value = *ipos;
      *opos++ = *ipos++;
    }

    value = _value;

    for (; ipos != iend; ++ipos, ++opos)
    {
      value += *ipos;
      *opos = value;
    }

    _value = value;

    seq._iseq._beg = seq._iseq._end;
    seq._obeg = opos;
  }

  void reduce(SequenceType& seq)
  {
    // assume _has_value

#if 0 // CONFIG_DEBUG
    printf("s [%x] reduce(%lu) %u\n",
	   (unsigned int)pthread_self(),
	   std::distance(this->_original_seq._obeg, seq._obeg),
	   _has_value ? _value : 0xffffffff);
#endif

    const ValueType value = _value;

    InputIteratorType ipos = seq._iseq._beg;
    OutputIteratorType opos = seq._obeg;

    for (; ipos < seq._iseq._end; ++ipos, ++opos)
      *opos += value;

    seq._iseq._beg = seq._iseq._end;
    seq._obeg = opos;
  }
};


// splitter

template<typename TaskType>
static void child_task(kaapi_task_t* , kaapi_stack_t*);

template<typename TaskType>
static int splitter
(
 kaapi_stack_t* victim_stack,
 kaapi_task_t* victim_task,
 int request_count,
 kaapi_request_t* request,
 void*
)
{
  typedef typename TaskType::ArgType WorkType;
  typedef typename WorkType::SequenceType SequenceType;

  WorkType* const victim_work =
    kaapi_task_getargst(victim_task, WorkType);

  const int saved_count = request_count;
  int replied_count = 0;

  if (request_count <= 0)
    return 0;

  const int work_size = victim_work->_remaining_seq.size();
  size_t par_size = work_size / (1 + (size_t)request_count);

  // todo: ParamType::ParSize
#define PAR_SIZE 2048
  if (par_size < PAR_SIZE)
  {
    request_count = work_size / PAR_SIZE - 1;
    par_size = PAR_SIZE;
  }

  // request_count can be <= 0
  for (; request_count > 0; ++request)
  {
    if (!kaapi_request_ok(request))
      continue ;

    // allocate on the thief stack
    kaapi_stack_t* const thief_stack = request->stack;
    kaapi_task_t* const thief_task = kaapi_stack_toptask(thief_stack);

    kaapi_task_initadaptive
    (
     thief_stack, thief_task,
     child_task<TaskType>, NULL,
     KAAPI_TASK_ADAPT_DEFAULT
    );
                
    WorkType* const thief_work = static_cast<WorkType*>
      (kaapi_stack_pushdata(thief_stack, sizeof(WorkType)));
    kaapi_task_setargs(thief_task, thief_work);

    victim_work->_remaining_seq.rsplit(thief_work->_remaining_seq, par_size);
    thief_work->_remaining_seq.empty_seq(thief_work->_processed_seq);

    TaskType::prepare(*thief_work, *victim_work);

    kaapi_stack_pushtask(thief_stack);

    kaapi_request_reply_head
    (
     victim_stack, victim_task,
     request, thief_stack,
     sizeof(WorkType), 1
    );

    --request_count;
    ++replied_count;
  }

  return saved_count - replied_count;
}


// task implems

template<typename WorkType>
struct BaseTask
{
  typedef WorkType ArgType;

  static void prepare(WorkType&, const WorkType&)
  {}

  static void entry(kaapi_task_t*, kaapi_stack_t*, WorkType*)
  {}
};


template<typename WorkType>
struct ReduceTask : BaseTask<WorkType>
{
  typedef typename WorkType::SequenceType SequenceType;

  static void prepare(WorkType& thief_work, const WorkType& victim_work)
  {
    // todo: work::prepare_reduce()
    thief_work._has_value = victim_work._has_value;
    if (victim_work._has_value == true)
      thief_work._value = victim_work._value;
  }

  static int _preempter
  (
   kaapi_stack_t* stack,
   kaapi_task_t* task,
   void* thief_data,
   void* victim_data,
   void*
  )
  {
    WorkType* const victim_work = static_cast<WorkType*>(victim_data);
    const WorkType* const thief_work = static_cast<WorkType*>(thief_data);

    victim_work->_processed_seq.join(thief_work->_processed_seq);
    victim_work->_remaining_seq = thief_work->_remaining_seq;

    return 1;
  }

  static void entry(kaapi_task_t* task, kaapi_stack_t* stack, WorkType* work)
  {
    typedef int (*kastl_preempter_t)
      (kaapi_stack_t*, kaapi_task_t*, void*, void*, void*);

    typedef int (*kastl_splitter_t)
      (kaapi_stack_t*, kaapi_task_t*, int, kaapi_request_t*, void*);

    kastl_splitter_t const _splitter = splitter< ReduceTask<WorkType> >;

    kastl_preempter_t const preempter = _preempter;

    SequenceType nseq;

  redo_work:
    while (work->_remaining_seq.is_empty() == false)
    {
#define NANO_SIZE ((size_t)4096)
      const size_t nano_size =
	std::min(work->_remaining_seq.size(), NANO_SIZE);
      work->_remaining_seq.split(nseq, nano_size);
      work->_processed_seq.join(nseq);

      kaapi_stealpoint(stack, task, _splitter, NULL);
      kaapi_stealbegin(stack, task, _splitter, NULL);
      work->reduce(nseq);
      kaapi_stealend(stack, task);
    }

    if (kaapi_preempt_nextthief(stack, task, NULL, preempter, work, NULL))
      goto redo_work;
  }
};


template<typename WorkType>
struct SequentialTask : BaseTask<WorkType>
{
  static void prepare(WorkType& thief_work, const WorkType& victim_work)
  {
    // todo: work::prepare_sequential()
    thief_work.prepare();
  }

  static int _preempter
  (
   kaapi_stack_t* stack,
   kaapi_task_t* task,
   void* thief_data,
   void* victim_data,
   void*
  )
  {
    typedef typename WorkType::SequenceType SequenceType;
  
    WorkType* const victim_work = static_cast<WorkType*>(victim_data);
    const WorkType* const thief_work = static_cast<WorkType*>(thief_data);

    // dont fork if nothing to reduce
    if (thief_work->_processed_seq.is_empty() == false)
    {
      kaapi_task_t* const new_task = kaapi_stack_toptask(stack);

      kaapi_task_initadaptive
      (
       stack, new_task,
       child_task< ReduceTask<WorkType> >,
       NULL, KAAPI_TASK_ADAPT_DEFAULT
      );

      WorkType* const new_work = static_cast<WorkType*>
	(kaapi_stack_pushdata(stack, sizeof(WorkType)));
      kaapi_task_setargs(new_task, new_work);

      // reduce the processed sequence
      new_work->_remaining_seq = thief_work->_processed_seq;
      new_work->_remaining_seq.empty_seq(new_work->_processed_seq);

#if CONFIG_DEBUG
      new_work->_original_seq = thief_work->_original_seq;
      new_work->_is_master = false;
#endif

      // todo: work::prepare_reduction
      new_work->_remaining_seq._obeg -= new_work->_remaining_seq.size();

      // todo: work::prepare_reduction
      new_work->_has_value = victim_work->_has_value;
      if (victim_work->_has_value)
	new_work->_value = victim_work->_value;

      kaapi_stack_pushtask(stack);
    }

    // join the processed seq, continue the remaining
    victim_work->_remaining_seq = thief_work->_remaining_seq;
    victim_work->_processed_seq.join(thief_work->_processed_seq);

    // todo: work::preempt_thief() here, not above
    if (thief_work->_has_value == true)
    {
      if (victim_work->_has_value == true)
      {
	victim_work->_value += thief_work->_value;
      }
      else
      {
	victim_work->_value = thief_work->_value;
	victim_work->_has_value = true;
      }
    }

    return 1;
  }

  static void entry(kaapi_task_t* task, kaapi_stack_t* stack, WorkType* work)
  {
    // daouda Ps process

#if 0 // CONFIG_DEBUG
  printf("%c [%x] process_steal %lu\n",
	 work->_is_master ? 'm' : 's',
	 (unsigned int)pthread_self(),
	 work->_remaining_seq.size());
#endif

    typedef int (*kastl_preempter_t)
      (kaapi_stack_t*, kaapi_task_t*, void*, void*, void*);
    kastl_preempter_t const preempter = _preempter;

    typename WorkType::SequenceType remaining_seq =
      work->_remaining_seq;

#define MACRO_SIZE ((size_t)16384)
    const size_t macro_size =
      std::min(remaining_seq.size(), MACRO_SIZE);
    remaining_seq.split(work->_remaining_seq, macro_size);

  redo_work:
    while (work->_remaining_seq.is_empty() == false)
      process_work(task, stack, work);

    // process preempted thief remaining seq
    if (kaapi_preempt_nextthief(stack, task, NULL, preempter, work, NULL))
      goto redo_work;

    // next macro iteration
    if (remaining_seq.is_empty() == true)
      return ;

    // push an adaptive task to continue the macro
    kaapi_task_t* const new_task = kaapi_stack_toptask(stack);

    kaapi_task_initadaptive
    (
     stack, new_task,
     child_task< SequentialTask<WorkType> >,
     NULL, KAAPI_TASK_ADAPT_DEFAULT
    );

    WorkType* const new_work = static_cast<WorkType*>
      (kaapi_stack_pushdata(stack, sizeof(WorkType)));
    kaapi_task_setargs(new_task, new_work);

    new_work->_remaining_seq = remaining_seq;
    remaining_seq.empty_seq(new_work->_processed_seq);

    // propagate sequential results
    // todo: work::prepare_sequential(WorkType&)
    new_work->_has_value = work->_has_value;
    if (work->_has_value == true)
      new_work->_value = work->_value;

    kaapi_stack_pushtask(stack);
  }
};


template<typename WorkType>
struct StealTask : BaseTask<WorkType>
{
  static void prepare(WorkType& thief_work, const WorkType& victim_work)
  {
    thief_work.prepare(victim_work);
  }

  static void entry(kaapi_task_t* task, kaapi_stack_t* stack, WorkType* work)
  {
    // daouda Pv process

    typedef typename WorkType::SequenceType SequenceType;
    process_work(task, stack, work);
    kaapi_preemptpoint(stack, task, NULL, work, sizeof(WorkType));
  }
};


// task helpers

template<typename TaskType>
static void child_task(kaapi_task_t* task, kaapi_stack_t* stack)
{
  typedef typename TaskType::ArgType ArgType;

  ArgType* const arg = kaapi_task_getargst(task, ArgType);
  TaskType::entry(task, stack, arg);
  kaapi_return_steal(stack, task, arg, sizeof(ArgType));
}

template<typename WorkType>
static void root_task(kaapi_task_t* task, kaapi_stack_t* stack)
{
  WorkType* const work = kaapi_task_getargst(task, WorkType);
  SequentialTask<WorkType>::entry(task, stack, work);
}


template<typename WorkType>
static void process_work
(
 kaapi_task_t* task,
 kaapi_stack_t* stack,
 WorkType* work
)
{
  typedef int (*kastl_splitter_t)
    (kaapi_stack_t*, kaapi_task_t*, int, kaapi_request_t*, void*);

  kastl_splitter_t const _splitter = splitter< StealTask<WorkType> >;

  typename WorkType::SequenceType nseq;

  while (true)
  {
    const size_t nano_size =
      std::min(NANO_SIZE, work->_remaining_seq.size());
    if (nano_size == 0)
      break;

    work->_remaining_seq.split(nseq, nano_size);

    // while extract_nano(nseq)
    kaapi_stealpoint(stack, task, _splitter, NULL);
    kaapi_stealbegin(stack, task, _splitter, NULL);
    work->process(nseq);
    kaapi_stealend(stack, task);

    // join the resulting sequence
    work->_processed_seq.join(nseq);
  }
}


template<typename WorkType>
static void push_sequential_task
(kaapi_stack_t* stack, kaapi_task_t* task, WorkType* work)
{
  kaapi_task_initadaptive
  (
   stack, task,
   root_task<WorkType>,
   static_cast<void*>(work),
   KAAPI_TASK_ADAPT_DEFAULT
  );
  kaapi_stack_pushtask(stack);
  kaapi_finalize_steal(stack, task);
}


template<typename WorkType>
static void compute(WorkType work)
{
  kaapi_stack_t* const stack = kaapi_self_stack();
  kaapi_task_t* const task = kaapi_stack_toptask(stack);
  kaapi_frame_t frame;

  kaapi_stack_save_frame(stack, &frame);
  push_sequential_task(stack, task, &work);
  kaapi_sched_sync(stack);
  kaapi_stack_restore_frame(stack, &frame);
}


template<typename InputIterator, typename OutputIterator>
static void kastl_partial_sum
(
 const InputIterator& ibeg,
 const InputIterator& iend,
 const OutputIterator& obeg
)
{
  typedef InOutSequence<InputIterator, OutputIterator> SequenceType;
  SequenceType seq(ibeg, iend, obeg);

  PrefixWork<SequenceType> work(seq);

  compute(work);
}


// main

#include <stdlib.h>
#include <sys/time.h>
#include <vector>

int main(int ac, char** av)
{
  std::vector<unsigned int> iv((ac == 2) ? atoi(av[1]) : 1000000);
  std::vector<unsigned int> ov0(iv.size());
  std::vector<unsigned int> ov1(iv.size());

  for (size_t i = 0; i < iv.size(); ++i)
    iv[i] = 1;

  typedef std::vector<unsigned int>::iterator IteratorType;

  struct timeval now[2];
  struct timeval diff[2];

  kastl_partial_sum(iv.begin(), iv.end(), ov0.begin());
  gettimeofday(&now[0], NULL);
  kastl_partial_sum(iv.begin(), iv.end(), ov0.begin());
  gettimeofday(&now[1], NULL);
  timersub(&now[1], &now[0], &diff[0]);

  std::partial_sum(iv.begin(), iv.end(), ov1.begin());
  gettimeofday(&now[0], NULL);
  std::partial_sum(iv.begin(), iv.end(), ov1.begin());
  gettimeofday(&now[1], NULL);
  timersub(&now[1], &now[0], &diff[1]);

  IteratorType beg0 = ov0.begin();
  IteratorType end0 = ov0.end();
  IteratorType beg1 = ov1.begin();

  bool is_valid = true;

  for (; beg0 != end0; ++beg0, ++beg1)
    if (*beg0 != *beg1)
    {
      printf("invalid @%lu\n", std::distance(ov0.begin(), beg0));
      is_valid = false;
      break;
    }

  printf("res %c k: %lf, s: %lf\n",
	 is_valid == true ? 'x' : '!',
	 (double)diff[0].tv_sec * 1000000.f + (double)diff[0].tv_usec,
	 (double)diff[1].tv_sec * 1000000.f + (double)diff[1].tv_usec);


  return 0;
}
