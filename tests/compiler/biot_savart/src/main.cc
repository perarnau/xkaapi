//
// Made by fabien le mentec <texane@gmail.com>
// 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.hh"
#include "x.hh"


// x to space translation
// 2d dimensions
// assume space has scaled down x dimensions
// objects have space coord

static inline unsigned int space_to_x(double n)
{
  return (unsigned int)(n * CONFIG_SPACE_SCALE);
}

static inline unsigned int space_to_x(unsigned int n)
{
  return n * CONFIG_SPACE_SCALE;
}

static inline unsigned int x_to_space(unsigned int n)
{
  return (double)n / (double)CONFIG_SPACE_SCALE;
}


// objects have space coords

typedef struct wire
{
  double x0, y0, x1, y1; // position
  double i; // current intensity
} wire_t;

typedef struct bfield
{
  double dimx, dimy; // dimension
  double norm; // for normalisation
  double* p; // data
} bfield_t;

static void zero_bfield(bfield_t& f)
{
  unsigned int xy = (unsigned int)f.dimx * (unsigned int)f.dimy;
  double* p = f.p;
  for (; xy; --xy, ++p) *p = 0;
  f.norm = 0;
}

static void init_bfield(bfield_t& f, double dimx, double dimy)
{
  f.dimx = dimx;
  f.dimy = dimy;

  f.p = (double*)::malloc((unsigned int)(dimy * dimx) * sizeof(double));
}

static void fini_bfield(bfield_t& f)
{
  free(f.p);
}

#pragma kaapi task \
  value(wx, dx) \
  value(wy, dy) \
  value(wi) \
  value(dw, wlen) \
  value(bx, by) \
  write(p)
static void integrate_over_wire
(
 double wx, double dx,
 double wy, double dy,
 double wi, // intensity
 double dw, double wlen,
 double bx, double by,
 double* p
)
{
  static const double mu0 = 4E-7 * M_PI;

  // precompte integral factor
  const double mui = (mu0 * wi) / (4. * M_PI);

  // integrate over the wire length
  double sum = 0;

  for (double d = 0; d < wlen; d += dw, wx += dx, wy += dy)
  {
    // r the distance from dl to measured point
    const double fu = bx - wx;
    const double bar = by - wy;
    const double rr = fu * fu + bar * bar;
    const double r = ::sqrt(rr);

    // exclude the dipole itself
    if (r < 1) continue ;

    // unit vector in the direction from the wire to the bpoint
    const double rhat[2] = { (bx - wx) / r, (by - wy) / r };

    // compute the dot product to get the cosa
    const double cosa = (dx * rhat[0] + dy * rhat[1]) / (r * dw);
    const double sina = 1 - cosa;

    // cross_product(dl, rhat)
    // a x b = |a| x |b| x sin(alpha)
    // where a and b 2 vectors, alpha the angle between them
    // since rhat is unit, only used to get the direction right
    const double cross = dw * r * sina;

    // integrate
    sum += cross / rr;
  }

  // assign bfield point
  *p += mui * sum;
}

static void compute_bfield(bfield_t& f, const wire_t& w)
{
  // compute the bfield produced by a wire w
  // biot-savart formula is used

  // precompute wire length
  const double wirew = w.x1 - w.x0;
  const double wireh = w.y1 - w.y0;
  const double wlen = ::sqrt(wirew * wirew + wireh * wireh);

  // dl{u, v} the current vector, stepped by dw
  double dl[2];
  static const double dw = .05;
  dl[0] = dw * (w.x1 - w.x0) / wlen;
  dl[1] = dw * (w.y1 - w.y0) / wlen;

  // foreach bfield point
  double* p = f.p;
  for (double bx = 0; bx < f.dimx; ++bx)
  {
    for (double by = 0; by < f.dimy; ++by, ++p)
    {
      integrate_over_wire
      (
       w.x0, dl[0],
       w.y0, dl[1],
       w.i,
       dw, wlen,
       bx, by,
       p
      );
    }
  }

#pragma kaapi barrier

  // update norm value
  unsigned int xy = f.dimx * f.dimy;
  for (p = f.p; xy; --xy, ++p) if (*p > f.norm) f.norm = *p;
}


static void normalize_bfield(bfield_t& f)
{
  // normalize
  unsigned int xy = (unsigned int)(f.dimx * f.dimy);
  for (double* p = f.p; xy; --xy, ++p) *p /= f.norm;
}


// global stuffs

static bfield_t g_bfield;
#define CONFIG_WIRE_COUNT 8
static wire_t g_wires[CONFIG_WIRE_COUNT];

static bool g_has_changed;

static void create_wire_loop(void)
{
  const unsigned int wire_count = CONFIG_WIRE_COUNT;
  const double inca = 2 * M_PI / wire_count;

  // loop center
  const double centerx = x_to_space(x_get_width() / 2);
  const double centery = x_to_space(x_get_height() / 2);

  // the loop radius
  static const double r = 10;

  // generate loop segments
  double a = 0;
  for (unsigned int i = 0; i < wire_count; ++i, a += inca)
  {
    g_wires[i].i = 1;

    // except for the first one
    if (i != 0)
    {
      g_wires[i].x0 = g_wires[i - 1].x1;
      g_wires[i].y0 = g_wires[i - 1].y1;
    }

    const double cosa = ::cos(a);
    const double sina = ::sin(a);

    g_wires[i].x1 = centerx + cosa * r;
    g_wires[i].y1 = centery + sina * r;
  }

  // link the first one
  g_wires[0].x0 = g_wires[wire_count - 1].x1;
  g_wires[0].y0 = g_wires[wire_count - 1].y1;
}

static void create_wire_zigzag(void)
{
  int dir = 1;

  const double centy = x_to_space(x_get_height() / 2);

  const double step = (x_to_space(x_get_width()) - 100) / CONFIG_WIRE_COUNT;
  double x = 50;

  g_wires[0].x0 = x;
  g_wires[0].y0 = centy;

  for (unsigned int i = 0; i < CONFIG_WIRE_COUNT; ++i, dir *= -1, x += step)
  {
    if (i != 0)
    {
      g_wires[i].x0 = g_wires[i - 1].x1;
      g_wires[i].y0 = g_wires[i - 1].y1;
    }

    g_wires[i].x1 = x;
    g_wires[i].y1 = centy + 2 * dir;

    g_wires[i].i = 100;
  }
}

static void init_globals(void)
{
  const double spacew = x_to_space((unsigned int)x_get_width());
  const double spaceh = x_to_space((unsigned int)x_get_height());

  create_wire_loop();

  init_bfield(g_bfield, spacew, spaceh);
  zero_bfield(g_bfield);
#pragma kaapi parallel
  {
    for (unsigned int i = 0; i < CONFIG_WIRE_COUNT; ++i)
      compute_bfield(g_bfield, g_wires[i]);
  }
  normalize_bfield(g_bfield);

  g_has_changed = true;
}

static void fini_globals(void)
{
  fini_bfield(g_bfield);
}


// redraw the scene

static void draw_wire(const wire_t& w)
{
  // todo: use surface

  static const x_color_t* green_color = NULL;

  if (green_color == NULL)
  {
    static const unsigned char green_rgb[3] = { 0xa0, 0, 0 };
    x_alloc_color(green_rgb, &green_color);
  }

  const unsigned int x0 = space_to_x(w.x0);
  const unsigned int x1 = space_to_x(w.x1);
  const unsigned int y0 = space_to_x(w.y0);
  const unsigned int y1 = space_to_x(w.y1);

  for (unsigned int i = 0; i < CONFIG_SPACE_SCALE; ++i)
    x_draw_line(x0, y0 + i, x1, y1 + i, green_color);
}

static void draw_bfield(const bfield_t& f)
{
  static const x_color_t* field_color = NULL;

  unsigned char field_rgb[3] = { 0, 0, 0 };
  if (field_color == NULL)
    x_alloc_color(field_rgb, &field_color);

  // draw each point of the bfield (intensity normalized to 1.)
  const unsigned int dimx = (unsigned int)f.dimx;
  const unsigned int dimy = (unsigned int)f.dimy;
  double* p = f.p;
  for (unsigned int x = 0; x < dimx; ++x)
  {
    for (unsigned int y = 0; y < dimy; ++y, ++p)
    {
      const double b = *p;
      field_rgb[0] = (unsigned char)(b * 255.);
      field_rgb[1] = (unsigned char)(b * 255.);
      field_rgb[2] = (unsigned char)(b * 255.);
      x_remap_color(field_rgb, field_color);

      // compute graphic coords
      const unsigned int x_ = space_to_x(x);
      const unsigned int y_ = space_to_x(y);

      for (unsigned int i = 0; i < CONFIG_SPACE_SCALE; ++i)
	x_draw_line(x_, y_ + i, x_ + CONFIG_SPACE_SCALE, y_ + i, field_color);
    }
  }
}

static void redraw(void*)
{
  if (g_has_changed == true)
  {
    // redraw always
    // g_has_changed = false;
    draw_bfield(g_bfield);

    const unsigned int wire_count = CONFIG_WIRE_COUNT;
    for (unsigned int i = 0; i < wire_count; ++i)
      draw_wire(g_wires[i]);
  }
}

// x event handlers

static int on_event(const struct x_event* ev, void* arg)
{
  switch (x_event_get_type(ev))
  {
  case X_EVENT_TICK:
    redraw(arg);
    break;

  case X_EVENT_QUIT:
    x_cleanup();
    fini_globals();
    ::exit(-1);
    break;

  default:
    break;
  }

  return 0;
}


// main

int main(int ac, char** av)
{
  // trigger every 40ms
  if (x_initialize(CONFIG_TICK_MS) == -1)
    return -1;

  // after x
  init_globals();

  // loop until done
  x_loop(on_event, (void*)NULL);

  return 0;
}
