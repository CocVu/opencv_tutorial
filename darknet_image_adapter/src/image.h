#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

typedef struct {
	int w;
	int h;
  int c;
  float *data;
} image;
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image **load_alphabet();

//float get_pixel(image m, int x, int y, int c);
//float get_pixel_extend(image m, int x, int y, int c);
//void set_pixel(image m, int x, int y, int c, float val);
//void add_pixel(image m, int x, int y, int c, float val);
float bilinear_interpolate(image im, float x, float y, int c);

image get_image_layer(image m, int l);

void free_image(image m);
void test_resize(char *filename);
#endif

