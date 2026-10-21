#include <stdio.h>
#include <stdlib.h>

#include "../MPI_parallelization/convo.c"

static int failures = 0;

#define CHECK_EQUAL(label, expected, actual)                                  \
  do {                                                                        \
    int expected_value = (expected);                                          \
    int actual_value = (actual);                                              \
    if (expected_value != actual_value) {                                     \
      fprintf(stderr, "%s: expected %d, got %d\n", label, expected_value,   \
              actual_value);                                                  \
      failures++;                                                             \
    }                                                                         \
  } while (0)

static void test_filters(void) {
  unsigned char values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  CHECK_EQUAL("average filter",
              5, filter(CONVOL_AVG1, values[0], values[1], values[2],
                        values[3], values[4], values[5], values[6],
                        values[7], values[8]));
  CHECK_EQUAL("weighted average",
              3, filter(CONVOL_AVG2, 0, 0, 0, 0, 9, 0, 0, 0, 0));
  CHECK_EQUAL("Laplacian edge filter",
              160, filter(CONVOL_PEAK, 0, 0, 0, 0, 10, 0, 0, 0, 0));
  CHECK_EQUAL("maximum-gradient filter",
              32, filter(CONVOL_STEP, 0, 0, 0, 0, 10, 2, 0, 8, 0));
  CHECK_EQUAL("median filter",
              5, filter(CONVOL_MEDIAN, 9, 1, 8, 2, 7, 3, 6, 4, 5));
  CHECK_EQUAL("zero denominator", 0, division(123, 0));
}

static void test_convolution(void) {
  unsigned char image[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  unsigned char single_pixel[1] = {42};

  CHECK_EQUAL("convolution status", 0, convolution(CONVOL_AVG1, image, 3, 3));
  CHECK_EQUAL("convolved center pixel", 5, image[4]);
  CHECK_EQUAL("top-left border remains unchanged", 1, image[0]);
  CHECK_EQUAL("bottom-right border remains unchanged", 9, image[8]);
  CHECK_EQUAL("single-pixel convolution status",
              0, convolution(CONVOL_MEDIAN, single_pixel, 1, 1));
  CHECK_EQUAL("single pixel remains unchanged", 42, single_pixel[0]);
}

static void test_argument_parsing(void) {
  int value = -1;

  CHECK_EQUAL("parse zero", 1, parse_nonnegative_int("0", &value));
  CHECK_EQUAL("parsed zero value", 0, value);
  CHECK_EQUAL("parse maximum", 1, parse_nonnegative_int("2147483647", &value));
  CHECK_EQUAL("parse negative", 0, parse_nonnegative_int("-1", &value));
  CHECK_EQUAL("parse trailing characters",
              0, parse_nonnegative_int("12x", &value));
  CHECK_EQUAL("parse overflow",
              0, parse_nonnegative_int("999999999999999999999", &value));
  CHECK_EQUAL("parse valid filter", 1, parse_filter_arg("4", &value));
  CHECK_EQUAL("parse invalid filter", 0, parse_filter_arg("5", &value));
}

int main(void) {
  (void)usage;
  test_filters();
  test_convolution();
  test_argument_parsing();

  if (failures != 0) {
    fprintf(stderr, "%d test(s) failed.\n", failures);
    return EXIT_FAILURE;
  }

  puts("All CPU convolution tests passed.");
  return EXIT_SUCCESS;
}
