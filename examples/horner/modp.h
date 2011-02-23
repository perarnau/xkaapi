#ifndef MODP_H_INCLUDED
# define MODP_H_INCLUDED

/* modp arithmetics
 */

static inline unsigned long modp
(unsigned long n)
{ return n % 1001; }

static inline unsigned long mul_modp
(unsigned long a, unsigned long b)
{
  /* (a * b) mod p */
  return modp(a * b);
}

static inline unsigned long add_modp
(unsigned long a, unsigned long b)
{
  /* (a + b) mod p */
  return modp(a + b);
}

static inline unsigned long pow_modp
(unsigned long a, unsigned long n)
{
  /* (x^y) mod p */

  unsigned long an;

  if (n == 0) return 1;
  else if (n == 1) return a;

  /* odd */
  if (n & 1)
    return mul_modp(a, pow_modp(a, n - 1));
  
  /* even */
  an = pow_modp(a, n / 2);
  return mul_modp(an, an);
}

static inline unsigned long axnb_modp
(unsigned long a, unsigned long x, unsigned long n, unsigned long b)
{
  /* (a * x^n + b) mod p */
  return add_modp(mul_modp(a, pow_modp(x, n)), b);
}

static inline unsigned long axb_modp
(unsigned long a, unsigned long x, unsigned long b)
{
  /* (a * x + b) mod p */
  return add_modp(mul_modp(a, x), b);
}

#endif /* ! MODP_H_INCLUDED */
