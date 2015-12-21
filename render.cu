#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <cuda.h>
#include <curand_kernel.h>

struct vec3 {
  float x, y, z;
  __device__ __host__ vec3(float x = 0, float y = 0, float z = 0):x(x),y(y),z(z) {}
  __device__ __host__ vec3 operator+(const vec3& a) const { return vec3(x + a.x, y + a.y, z + a.z); }
  __device__ __host__ vec3 operator-(const vec3& a) const { return vec3(x - a.x, y - a.y, z - a.z); }
  __device__ __host__ float operator*(const vec3& a) const { return x * a.x + y * a.y + z * a.z; }
  __device__ __host__ vec3 operator%(const vec3& a) const { return vec3(y * a.z - z * a.y, x * a.z - z * a.x, x * a.y - y - a.x); }
  __device__ __host__ vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
  __device__ __host__ vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
  __device__ __host__ vec3 operator~() const { return (*this) / sqrt(*this * *this); }
  __device__ __host__ float operator+() const { return sqrt(*this * *this); }
  __device__ __host__ vec3 operator&&(const vec3& a) const { return vec3(x * a.x, y * a.y, z * a.z); }
  __device__ __host__ vec3 operator!() const { return vec3(x < 0 ? -x : x, y < 0 ? -y : y, z < 0 ? -z : z); }

  void print(const char* s = "") const { printf("%s %lg %lg %lg\n", s, x, y, z); }
};

struct ray {
  vec3 p, d, w;
  __device__ __host__ ray(const vec3& p = vec3(), const vec3& d = vec3()):p(p),d(d),w(vec3(1, 1, 1)) {}
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
  vec3 col;
  bool emit;
  obj():type(NOTHING){}
};

#ifdef USE_CUDA
__device__
#endif
float intersect_sphere(const sphere& s, const ray& r, vec3* n) {
  float a = r.d * r.d;
  float b = r.d*(r.p - s.p)*2;
  float c = (r.p - s.p)*(r.p - s.p)-s.r*s.r;
  float d = b*b - 4*a*c;
  if (d < 0) return -1;
  float t0 = (-b-sqrt(d))/2/a;
  float t1 = (-b+sqrt(d))/2/a;
  float t;
  if (t0 < 0) t = t1;
  else t = t0;
  vec3 p = r.p + r.d * t;
  *n = ~(p - s.p);
  return t;
}

#ifdef USE_CUDA
__device__
#endif
float intersect_plane(const plane& p, const ray& r, vec3* n) {
  *n = p.n;
  return (p.p - r.p) * p.n / (r.d * p.n);
}

#ifdef USE_CUDA
__device__
#endif
float intersect(const obj& o, const ray& r, vec3* n) {
  switch (o.type) {
    case obj::PLANE: return intersect_plane(o.p, r, n);
    case obj::SPHERE: return intersect_sphere(o.s, r, n);
  }
  return -1;
}

ray rays[600 * 600];
obj objs[8];
ray* rays_;
obj* objs_;
vec3* screen_;

__global__ void setup_kernel(curandState *state) {
#ifdef USE_CUDA
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(1234, i, 0, &state[i]);
#endif
}

curandState *r_state;
__device__ __host__ vec3 rv(curandState* state, const vec3& n) {
  float x, y, z, d;
  vec3 t;
#ifdef USE_CUDA
  int i = threadIdx.x + blockDim.x * blockIdx.x;
#endif
  do {
#ifdef USE_CUDA
    x = curand_uniform(&state[i]) * 2 - 1;
    y = curand_uniform(&state[i]) * 2 - 1;
    z = curand_uniform(&state[i]) * 2 - 1;
#else
    x = rand() * 1. / RAND_MAX * 2 - 1;
    y = rand() * 1. / RAND_MAX * 2 - 1;
    z = rand() * 1. / RAND_MAX * 2 - 1;
#endif
    d = sqrt(x*x + y*y + z*z);
    t = vec3(x/d, y/d, z/d);
  } while (d > 1 && n*t < 0);
  return t;
}

#ifdef USE_CUDA
__global__
#endif
void trace(curandState *state, ray* rays, obj* objs, vec3* screen, int i_) {
#ifdef USE_CUDA
  int i = threadIdx.x + blockDim.x * blockIdx.x;
#else
  int i = i_;
#endif
  if (i > 600 * 600) return;
  if (+rays[i].w < 0.01) return;
  float t = -1;
  obj* o = NULL;;
  vec3 n;
  for (int j = 0; j < 8; j++) {
    vec3 n_ = vec3(0, 0, 0);
    float t0 = intersect(objs[j], rays[i], &n_);
    if (t0 > 0.001 && t < 0 || t0 > 0.001 && t0 < t) {
      t = t0;
      o = &objs[j];
      n = n_;
    }
  }
    /* if (t < 0) screen[i] = vec3(); */
    /* else screen[i] = !n; //vec3(1/t, 1/t, 1/t); */
    /* return; */
  if (t > 0.001) {
    /* screen[i] = (*o).col; */
    /* return; */
    if (n * rays[i].d > 0) {
      rays[i].w = vec3();
      //screen[i] = vec3(1, 0, 1);
      return;
    } else {
      // screen[i] = (*o).col;
    }
    if ((*o).emit) {
      screen[i] = screen[i] + ((*o).col && rays[i].w);
      rays[i].w = vec3();
      return;
    }
    float diff = n * rays[i].d * -1;
    diff = diff > 0 ? diff : 0;
    rays[i].p = rays[i].p + rays[i].d * t;
    rays[i].d = rv(state, n);
    rays[i].w = (rays[i].w && (*o).col) * diff * 0.7;
  } else {
    rays[i].w = vec3();
  }
}

__device__ __host__ float r(curandState* state) {
#ifdef USE_CUDA
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  return curand_uniform(&state[i]) * 2 - 1;
#else
  return 1. * rand() / RAND_MAX * 2 - 1;
#endif
}

#ifdef USE_CUDA
__global__
#endif
void genRays(curandState* state, ray* rays, int i_) {
#ifdef USE_CUDA
  int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
  int i = i_;
#endif
  if (i > 600 * 600) return;

  vec3 eye(0, 0, 5);
  vec3 lookat(0, 0, 2);
  vec3 up(0, 1, 0);
  vec3 right(1, 0, 0);
  // up = ~((eye - lookat) % right);
  int x = i % 600;
  int y = i / 600;
  vec3 t = lookat + right * ((x - 300 + r(state)) / 300.) + up * ((y - 300 + r(state)) / 300.);
  rays[i] = ray(eye, ~(t - eye));
}

void err(const char* s = "") {
  // printf("%s: %s\n", s, cudaGetErrorString(cudaGetLastError()));
}

void render(float* s) {
  vec3* screen = (vec3*) s;
  for (int i = 0; i < 360000; i++) {
    screen[i] = screen[i] * 0.99;
  }

#ifdef USE_CUDA
  cudaMemcpy(screen_, screen, 600*600*sizeof(vec3), cudaMemcpyHostToDevice); err("memcpy screen");
#endif

  for (int i = 0; i < 10; i++) {
#ifdef USE_CUDA
    genRays<<<600, 600>>>(r_state, rays_, 0);
#else
    for (int k = 0; k < 360000; k++) genRays(r_state, rays_, k);
#endif

    for (int j = 0; j < 6; j++) {
#ifdef USE_CUDA
      trace<<<600, 600>>>(r_state, rays_, objs_, screen_, 0); err("call trace");
#else
    for (int k = 0; k < 360000; k++) trace(r_state, rays_, objs, screen, k);
#endif
    }
  }

#ifdef USE_CUDA
  cudaMemcpy(screen, screen_, 600*600*sizeof(vec3), cudaMemcpyDeviceToHost); err("back memcpy screen");
#endif

  float ml = 0;
  for (int i = 0; i < 360000; i++) {
    float l = .21 * screen[i].x + 0.71 * screen[i].y + .07 * screen[i].z;
    ml = l > ml ? l : ml;
  }
  ml = ml / (1 + ml);
  for (int i = 0; i < 360000; i++) {
    screen[i] = screen[i];;
    /* screen[i].x = screen[i].x > 1 ? 1 : screen[i].x; */
    /* screen[i].y = screen[i].y > 1 ? 1 : screen[i].y; */
    /* screen[i].z = screen[i].z > 1 ? 1 : screen[i].z; */
  }
}

void init() {
#ifdef USE_CUDA
  cudaMalloc(&r_state, 600*600*sizeof(curandState)); err("malloc rand");
  cudaMalloc(&rays_, 600*600*sizeof(ray)); err("malloc rays");
  cudaMalloc(&objs_, 8*sizeof(obj)); err("malloc objs");
  cudaMalloc(&screen_, 600*600*sizeof(vec3)); err("malloc screen");
#else
  rays_ = (ray*)malloc(600*600*sizeof(ray));
#endif

  setup_kernel<<<600, 600>>>(r_state); err("call rand_setup");
  objs[0].type = obj::SPHERE;
  objs[0].col = vec3(1, 1, 1);
  objs[0].s.p = vec3(0.5, -0.5, 0);
  objs[0].s.r = 0.4;

  objs[1].type = obj::SPHERE;
  objs[1].s.p = vec3(0, 0.85, -0.85);
  objs[1].s.r = 0.1;
  objs[1].emit = true;
  objs[1].col = vec3(0.5, 0.5, 0.5);

  objs[2].type = obj::SPHERE;
  objs[2].col = vec3(1, 1, 1);
  objs[2].s.p = vec3(-0.5, -0.5, 0);
  objs[2].s.r = 0.4;

  objs[3].type = obj::PLANE;
  objs[3].col = vec3(1, 1, 1);
  objs[3].p.p = vec3(0, -1, 0);
  objs[3].p.n = vec3(0, 1, 0);

  objs[4].type = obj::PLANE;
  objs[4].col = vec3(0, 1, 0);
  objs[4].p.p = vec3(-1, 0, 0);
  objs[4].p.n = vec3(1, 0, 0);

  objs[5].type = obj::PLANE;
  objs[5].col = vec3(1, 0, 0);
  objs[5].p.p = vec3(1, 0, 0);
  objs[5].p.n = vec3(-1, 0, 0);

  objs[6].type = obj::PLANE;
  objs[6].col = vec3(1, 1, 1);
  objs[6].p.p = vec3(0, 1, 0);
  objs[6].p.n = vec3(0, -1, 0);

  objs[7].type = obj::PLANE;
  objs[7].col = vec3(1, 1, 1);
  objs[7].p.p = vec3(0, 0, -1);
  objs[7].p.n = vec3(0, 0, 1);

#ifdef USE_CUDA
  cudaMemcpy(objs_, objs, 8*sizeof(obj), cudaMemcpyHostToDevice); err("memcpy objs");
#endif
}
