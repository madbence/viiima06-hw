#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

#define GLSL(src) "#version 150 core\n" #src

struct vec3 {
  float x, y, z;
  vec3(float x = 0, float y = 0, float z = 0):x(x),y(y),z(z) {}
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
    -1,  1, 0, 0,
     1,  1, 1, 0,
    -1, -1, 0, 1,
    -1, -1, 0, 1,
     1,  1, 1, 0,
     1, -1, 1, 1
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

  int i = 0;
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
