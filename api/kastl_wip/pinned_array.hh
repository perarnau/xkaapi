#ifndef PINNED_ARRAY_HH_INCLUDED
# define PINNED_ARRAY_HH_INCLUDED



#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>



template<typename T>
class pinned_array
{
  size_t _max_size;
  size_t _cur_size;
  T* _data;
  bool _is_mlocked;

  inline void reset()
  {
    _data = NULL;
    _max_size = 0;
    _cur_size = 0;
    _is_mlocked = false;
  }

  bool allocate(size_t size)
  {
    const size_t typed_size = size * sizeof(T);

    _data = static_cast<T*>(malloc(typed_size));
    if (_data == NULL)
      return false;

    if (mlock((void*)_data, typed_size))
    {
      // do not fail but warn
      perror("mprobe");
    }
    else
    {
      _is_mlocked = true;
    }

    _max_size = size;
    _cur_size = size;

    return true;
  }

  void destroy()
  {
    if (_data == NULL)
      return ;

    if (_is_mlocked == true)
      munlock(_data, _max_size * sizeof(T));

    free(_data);
    reset();
  }
  
public:

  typedef T* iterator;

  pinned_array() : _max_size(0), _cur_size(0), _data(NULL)
  { }

  ~pinned_array()
  {
    destroy();
  }

  void resize(size_t size)
  {
    if (_max_size < size)
    {
      destroy();
      allocate(size);
    }

    _cur_size = size;
  }

  inline size_t size() const { return _cur_size; }

  inline T& operator[](size_t i) { return _data[i]; }
  inline T& operator[](int i) { return _data[i]; }

  inline T* begin() { return _data; }
  inline T* end() { return _data + _cur_size; }

};



#endif // ! PINNED_ARRAY_HH_INCLUDED
