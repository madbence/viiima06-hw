#include <stdio.h>
#include "render.h"

float screen[360000*3];

int main() {
  init();
  for (int i = 0; i < 20; i++) {
    render(screen);
  }
  printf("P6\n 600 600\n");
  for (int y = 0; y < 600; y++) {
    for (int x = 0; x < 600; x++) {
      for (int c = 0; c < 3; c++) {
        float f = screen[y * 600 * 3 + x * 3 + c];
        f = f > 1 ? 1 : f < 0 ? 0 : f;
        printf("%d ", (int)(f * 255));
      }
    }
    printf("\n");
  }
  return 0;
}
