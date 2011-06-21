#include <stdio.h>
#include <stdlib.h>


/* define the hist level too. assume <= 32. */
#define CONFIG_DATA_LOG2 12
/* dont touch */
#define CONFIG_DATA_DIM (1 << CONFIG_DATA_LOG2)
#define CONFIG_BLOCK_DIM (CONFIG_DATA_DIM / 4)


/* histogram reduction */

#pragma kaapi declare			\
  reduction(hist_redop: reduce_hist)	\
  identity(init_hist)

static void init_hist(unsigned int* p)
{
  unsigned int i;
  for (i = 0; i < CONFIG_DATA_DIM; ++i, ++p) *p = 0;
}

__attribute__((unused))
static void reduce_hist(unsigned int* lhs, const unsigned int* rhs)
{
  unsigned int i;
  for (i = 0; i < CONFIG_DATA_DIM; ++i, ++lhs, ++rhs) *lhs += *rhs;
}

#pragma kaapi task			\
  value(dim, lda)			\
  read(data{lda = lda; [dim][dim]})	\
  reduction(hist_redop: hist)
static void compute_block_hist
(
 const unsigned int* data,
 unsigned int dim,
 unsigned int lda,
 unsigned int* hist
)
{
  unsigned int x, y;
  for (y = 0; y < dim; ++y, data += lda)
    for (x = 0; x < dim; ++x, ++data)
      ++hist[*data];
}

static void compute_hist
(
 const unsigned int* data,
 unsigned int dim,
 unsigned int* hist
)
{
  static const unsigned int lda = CONFIG_DATA_DIM;
  static const unsigned int block_size =
    CONFIG_BLOCK_DIM * CONFIG_BLOCK_DIM;

  const unsigned int block_count = dim / CONFIG_BLOCK_DIM;
  unsigned int i, j;

  init_hist(hist);

  for (i = 0; i < block_count; ++i)
  {
    for (j = 0; j < block_count; ++j)
    {
      /* compute start of block */
      const unsigned int* const p =
      	data + i * block_size + j * CONFIG_BLOCK_DIM;
      compute_block_hist(p, CONFIG_BLOCK_DIM, lda, hist);
    }
  }
}


/* data helpers */

static unsigned int* gen_data(unsigned int dim)
{
  const unsigned int size = dim * dim;
  unsigned int* const data = malloc(size * sizeof(unsigned int));

  /* arrange so that each value appears dim times */
  unsigned int i;
  for (i = 0; i < size; ++i) data[i] = i & (dim - 1);
  return data;
}

static int check_hist(const unsigned int* hist)
{
  unsigned int i;
  for (i = 0; i < CONFIG_DATA_DIM; ++i)
    if (hist[i] != CONFIG_DATA_DIM) return -1;
  return 0;
}


/* main */

int main(int ac, char** av)
{
  static unsigned int hist[CONFIG_DATA_DIM];
  unsigned int* const data = gen_data(CONFIG_DATA_DIM);
  int err;

  if (data == NULL) return -1;

#pragma kaapi parallel
  compute_hist(data, CONFIG_DATA_DIM, hist);

  free(data);

  err = check_hist(hist);
  if (err == -1) printf("invalid\n");

  return err;
}
