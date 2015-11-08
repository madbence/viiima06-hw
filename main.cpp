#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <cmath>

#define GLSL(src) "#version 150 core\n" #src

struct vec3 {
  float x, y, z;
  vec3(float x = 0, float y = 0, float z = 0):x(x),y(y),z(z) {}
  vec3 operator+(const vec3& a) const { return vec3(x + a.x, y + a.y, z + a.z); }
  vec3 operator-(const vec3& a) const { return vec3(x - a.x, y - a.y, z - a.z); }
  float operator*(const vec3& a) const { return x * a.x + y * a.y + z * a.z; }
  vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
  vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
  vec3 operator~() const { return (*this) / sqrt(*this * *this); }
  void print(const char* s = "") const { printf("%s %lg %lg %lg\n", s, x, y, z); }
};

vec3 screen[600 * 600];

GLuint createShader(const char** source, GLenum type) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, source, NULL);
  glCompileShader(shader);
  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  char buffer[512];
  glGetShaderInfoLog(shader, 512, NULL, buffer);
  printf("Shader compiled: %s\n", buffer);
  return shader;
}

struct ray {
  vec3 p, d;
  ray(const vec3& p = vec3(), const vec3& d = vec3()):p(p),d(d) {}
};

struct plane {
  vec3 p, n;
};

struct sphere {
  vec3 p;
  float r;
};

struct obj {
  enum {PLANE, SPHERE, NOTHING} type;
  union {
    plane p;
    sphere s;
  };
  obj():type(NOTHING){}
};

float intersect_sphere(const sphere& s, const ray& r) {
  float a = r.d * r.d;
  float b = r.d*(r.p - s.p)*2;
  float c = (r.p - s.p)*(r.p - s.p)-s.r*s.r;
  float d = b*b - 4*a*c;
  if (d < 0) return -1;
  return (-b-sqrt(d))/a/2;
}

float intersect_plane(const plane& p, const ray& r) {
  return -1;
}

float intersect(const obj& o, const ray& r) {
  switch (o.type) {
    case obj::PLANE: return intersect_plane(o.p, r);
    case obj::SPHERE: return intersect_sphere(o.s, r);
  }
  return -1;
}

ray rays[600 * 600];
obj objs[2];

void trace(int i) {
  float t = -1;
  for (int j = 0; j < 2; j++) {
    float t0 = intersect(objs[j], rays[i]);
    t = t0 > t ? t0 : t;
  }
  if (t < 0) return;
  screen[i] = 1;
}

void render() {
  vec3 eye(0, 0, 5);
  vec3 lookat(0, 0, 0);
  vec3 up(0, 1, 0);
  vec3 right(1, 0, 0);

  for (int x = 0; x < 600; x++) {
    for (int y = 0; y < 600; y++) {
      vec3 t = lookat + right * ((x - 300) / 300.) + up * ((y - 300) / 300.);
      rays[y * 600 + x] = ray(eye, ~(t - eye));
    }
  }

  for (int i = 0; i < 600 * 600; i++) {
    trace(i);
  }
}

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  GLFWwindow* window = glfwCreateWindow(600, 600, "viiima06", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  glewInit();

  float vs[] = {
    -1,  1, 0, 1,
     1,  1, 1, 1,
    -1, -1, 0, 0,
    -1, -1, 0, 0,
     1,  1, 1, 1,
     1, -1, 1, 0
  };

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vs), vs, GL_STATIC_DRAW);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  const char* vertexShaderSource = GLSL(
    in vec2 pos;
    in vec2 tex;
    out vec2 Tex;
    void main() {
      gl_Position = vec4(pos, 0, 1);
      Tex = tex;
    }
  );
  const char* fragmentShaderSource = GLSL(
    in vec2 Tex;
    out vec4 outColor;
    uniform sampler2D sampler;
    void main() {
      outColor = texture(sampler, Tex);
    }
  );
  GLuint vertexShader = createShader(&vertexShaderSource, GL_VERTEX_SHADER);
  GLuint fragmentShader = createShader(&fragmentShaderSource, GL_FRAGMENT_SHADER);
  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glBindFragDataLocation(shaderProgram, 0, "outColor");
  glLinkProgram(shaderProgram);
  glUseProgram(shaderProgram);

  GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
  GLint texAttrib = glGetAttribLocation(shaderProgram, "tex");
  glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(posAttrib);
  glEnableVertexAttribArray(texAttrib);

  objs[0].type = obj::SPHERE;
  objs[0].s.p = vec3(0, 0, 0);
  objs[0].s.r = 0.5;

  objs[1].type = obj::SPHERE;
  objs[1].s.p = vec3(1, 1, 0);
  objs[1].s.r = 0.1;

  int i = 0;
    render();
  while(!glfwWindowShouldClose(window))
  {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 600, 600, 0, GL_RGB, GL_FLOAT, screen);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
  return 0;
}
