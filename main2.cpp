#include <stdio.h>
#include "render.h"

float screen[360000*3];

int main() {
  init();
  for (int i = 0; i < 10; i++) {
    render(screen);
  }
  printf("P3\n600 600 255\n");
  for (int y = 599; y >= 0; y--) {
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
