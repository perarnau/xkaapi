#pragma kaapi task read(key) read( [begin..end] ) write(index)
void search( long key, const long* begin, const long* end, long** index)
{
  size_t size = (end-begin)/2;
  if (end-begin <2)
  {
    if (*begin == key) *index = begin;
  }
  else 
  {
    /* simple recursive for_each */
    size_t med = (end-begin)/2;
    for_each( begin, begin+med, op);
    for_each( begin+med, end, op);
  }
}
