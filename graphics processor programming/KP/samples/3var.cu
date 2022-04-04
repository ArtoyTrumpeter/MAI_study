#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <vector>
#include <math.h>

// 3 вар Тетраэр, Гексаэдр, Икосаэдр

#define CSC(call)                                                 \
    do                                                            \
    {                                                             \
        cudaError_t res = call;                                   \
        if (res != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(res)); \
            exit(0);                                              \
        }                                                         \
    } while (0)

using namespace std;

struct vec3 {
    float x;
    float y;
    float z;
};

struct vec4 {
    float diffus;
    float spec;
    float refl;
    float refr;
};

struct material {
    float refractive_index;
    float specular_exponent;
    uchar4 color;
    vec4 albedo;
};

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    material mat;
    vec3 normal;
};

struct Light {
    vec3 pos;
    float intensity;
    vec3 clr;
};

__host__ __device__ float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(const vec3 &a, const vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ vec3 norm(const vec3 &v) {
    float l = sqrt(dot(v, v));
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ vec3 diff(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ vec3 add(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ vec3 mult(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

__host__ __device__ vec3 mult_const(const vec3 &a, float c) {
    return {a.x * c, a.y * c, a.z * c};
}

__host__ __device__ vec3 get_norm(const vec3 &a, const vec3 &b, const vec3 &c) {
    return norm(prod(diff(b, a), diff(c, a)));
}

__host__ __device__ int closest_trig(const vec3 &pos, const vec3 &dir, float &ts_min, trig *trigs, int n) {
    int k_min = -1;
    for (int k = 0; k < n; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
            continue;
        }
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if ((u < 0.0) || (u > 1.0)) {
            continue;
        }
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if ((v < 0.0) || (v + u > 1.0)) {
            continue;
        }
        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
            continue;
        }
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    return k_min;
}

__host__ __device__ vec3 reflect(const vec3 &I, const vec3 &N) {
    return diff(I, mult_const(mult_const(N, 2), dot(I, N)));
}

__host__ __device__ vec3 refract(const vec3 &I, const vec3 &N, const float eta_t, const float eta_i = 1.0) {
    float cosi = -max(-1.0, min(1.0, dot(I, N)));
    if (cosi < 0) {
        return refract(I, mult_const(N, -1), eta_i, eta_t);
    }
    float eta = eta_i / eta_t;
    float k = abs(1 - eta * eta * (1 - cosi * cosi));
    return k < 0 ? vec3{10, 0, 0} : norm(add(mult_const(I, eta), mult_const(N, (eta * cosi - sqrt(k)))));
}

__host__ __device__ uchar4 get_texture_clr(uchar4 *floor, float x, float y, int w, int floor_w) {
    x = (x / floor_w * w + w / 2);
    y = (y / floor_w * w + w / 2);
    return floor[(int)x * w + (int)y];
}

void floor_painting(uchar4 *floor, vec3 color, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        floor[i] = make_uchar4(color.x * floor[i].x, color.y * floor[i].y, color.z * floor[i].z, floor[i].w);
    }
}

void Tetrahedron(float radius, float c_x, float c_y, float c_z, const material &mat, vector<trig> &figures) {
    vec3 point_a = {radius + c_x, radius + c_y, radius + c_z};
    vec3 point_b = {radius + c_x, -radius + c_y, -radius + c_z};
    vec3 point_c = {-radius + c_x, radius + c_y, -radius + c_z};
    vec3 point_d = {-radius + c_x, -radius + c_y, radius + c_z};

    figures.push_back({point_a, point_b, point_d, mat});
    figures.push_back({point_a, point_c, point_d, mat});
    figures.push_back({point_b, point_c, point_d, mat});
    figures.push_back({point_a, point_b, point_c, mat});
}

void Hexahedron(float radius, float c_x, float c_y, float c_z, const material &mat, vector<trig> &figures) {
    vec3 point_a = {-radius + c_x, -radius + c_y, -radius + c_z};
    vec3 point_b = {-radius + c_x, -radius + c_y,  radius + c_z};
    vec3 point_c = {-radius + c_x,  radius + c_y, -radius + c_z};
    vec3 point_d = {-radius + c_x,  radius + c_y,  radius + c_z};
    vec3 point_e = { radius + c_x, -radius + c_y, -radius + c_z};
    vec3 point_f = { radius + c_x, -radius + c_y,  radius + c_z};
    vec3 point_g = { radius + c_x,  radius + c_y, -radius + c_z};
    vec3 point_h = { radius + c_x,  radius + c_y,  radius + c_z};
 
    figures.push_back({point_a, point_b, point_d, mat});
    figures.push_back({point_a, point_c, point_d, mat});
    figures.push_back({point_b, point_f, point_h, mat});
    figures.push_back({point_b, point_d, point_h, mat});
    figures.push_back({point_e, point_f, point_h, mat});
    figures.push_back({point_e, point_g, point_h, mat});
    figures.push_back({point_a, point_e, point_g, mat});
    figures.push_back({point_a, point_c, point_g, mat});
    figures.push_back({point_a, point_b, point_f, mat});
    figures.push_back({point_a, point_e, point_f, mat});
    figures.push_back({point_c, point_d, point_h, mat});
    figures.push_back({point_c, point_g, point_h, mat});
}

void Icosahedron(float radius, float c_x, float c_y, float c_z, const material &mat, vector<trig> &figures) {
    vec3 point_a = {c_x, -radius + c_y, radius + c_z};
    vec3 point_b = {c_x,  radius + c_y, radius + c_z};
    vec3 point_c = {c_x - radius, c_y, radius + c_z};
    vec3 point_d = {c_x + radius, c_y, radius + c_z};
    vec3 point_e = {c_x - radius, radius + c_y, c_z};
    vec3 point_f = {c_x + radius, radius + c_y, c_z};
    vec3 point_g = {c_x + radius, -radius + c_y, c_z};
    vec3 point_h = {c_x - radius, -radius + c_y, c_z};
    vec3 point_i = {c_x - radius, c_y, -radius + c_z};
    vec3 point_j = {c_x + radius, c_y, -radius + c_z};
    vec3 point_k = {c_x, -radius + c_y, -radius + c_z};
    vec3 point_l = {c_x,  radius + c_y, -radius + c_z};

    figures.push_back({ point_a,  point_b,  point_c, mat});
    figures.push_back({ point_b,  point_a,  point_d, mat});
    figures.push_back({ point_a,  point_c,  point_h, mat});
    figures.push_back({ point_c,  point_b,  point_e, mat});
    figures.push_back({ point_e,  point_b,  point_f, mat});
    figures.push_back({ point_g,  point_a,  point_h, mat});
    figures.push_back({ point_d,  point_a,  point_g, mat});
    figures.push_back({ point_b,  point_d,  point_f, mat});
    figures.push_back({ point_e,  point_f,  point_l, mat});
    figures.push_back({ point_g,  point_h,  point_k, mat});
    figures.push_back({ point_d,  point_g,  point_j, mat});
    figures.push_back({ point_f,  point_d,  point_j, mat});
    figures.push_back({ point_h,  point_c,  point_i, mat});
    figures.push_back({ point_c,  point_e,  point_i, mat});
    figures.push_back({ point_j,  point_k,  point_l, mat});
    figures.push_back({ point_k,  point_i,  point_l, mat});
    figures.push_back({ point_f,  point_j,  point_l, mat});
    figures.push_back({ point_j,  point_g,  point_k, mat});
    figures.push_back({ point_h,  point_i,  point_k, mat});
    figures.push_back({ point_i,  point_e,  point_l, mat});
}

void build_space(vector<trig> &figures, const vec3 &center1, const vec3 &center2, const vec3 &center3, const trig &fl1, const trig &fl2, const vec3 &rads, material *materials) {
    figures.push_back({fl1.a, fl1.b, fl1.c, materials[3], {0.0, 0.0, 1.0}});
    figures.push_back({fl2.a, fl2.b, fl2.c, materials[3], {0.0, 0.0, 1.0}});

    Tetrahedron(rads.x, center1.x, center1.y, center1.z, materials[0], figures);
    Hexahedron(rads.y, center2.x, center2.y, center2.z, materials[1], figures);
    Icosahedron(rads.z, center3.x, center3.y, center3.z, materials[2], figures);
}

__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, int depth, uchar4 *floor, int f_size, int f_txt_size, trig *trigs, Light *lights, int l_num, int max_deep, int n) {
    uchar4 reflect_color = {0, 0, 0, 0};
    uchar4 refract_color = {0, 0, 0, 0};

    if (depth > max_deep) {
        return {0, 0, 0, 0};
    }
    float ts_min = FLT_MAX;
    int k_min = closest_trig(pos, dir, ts_min, trigs, n);
    if (k_min == -1) {
        return {0, 0, 0, 0};
    }
    vec3 z = add(pos, mult_const(dir, ts_min));

    material cur_mat = trigs[k_min].mat;
    uchar4 clr = make_uchar4(cur_mat.color.x, cur_mat.color.y, cur_mat.color.z, cur_mat.color.w);

    if (k_min < 2) {
        clr = get_texture_clr(floor, z.x, z.y, f_txt_size, f_size);
    }

    vec3 N = trigs[k_min].normal;

    vec3 reflect_dir = norm(reflect(dir, N));
    vec3 refract_dir = refract(dir, N, cur_mat.refractive_index);
    vec3 z_near_rfl = dot(reflect_dir, N) < 0 ? diff(z, mult_const(N, 1e-6)) : add(z, mult_const(N, 1e-6));
    reflect_color = ray(z_near_rfl, reflect_dir, depth + 1, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep, n);

    if (refract_dir.x <= 1.0) {
        vec3 z_near_rfr = dot(norm(refract_dir), N) < 0 ? diff(z, mult_const(N, 1e-6)) : add(z, mult_const(N, 1e-6));
        refract_color = ray(z_near_rfr, norm(refract_dir), depth + 1, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep, n);
    }

    vec3 sum_clr = {0.0, 0.0, 0.0};
    float ambient_intens = 0.1;

    for (int i = 0; i < l_num; i++) {
        float diffuse_intens = 0.0;
        float specular_intens = 0.0;

        bool shadow = false;
        vec3 light_dir = norm(diff(lights[i].pos, z));
        vec3 to_light_dir = mult_const(light_dir, -1);

        ts_min = FLT_MAX;
        int l_min = closest_trig(lights[i].pos, to_light_dir, ts_min, trigs, n);
        if (l_min != k_min) {
            shadow = true;
        }

        if (!shadow) {
            diffuse_intens = lights[i].intensity * max(0.0, dot(light_dir, N));
            specular_intens = pow(max(0.0, dot(mult_const(reflect(mult_const(light_dir, -1), N), -1), dir)), (double)cur_mat.specular_exponent) * lights[i].intensity;
        }
        sum_clr.x += lights[i].clr.x * clr.x * diffuse_intens * cur_mat.albedo.diffus + 255.0 * specular_intens * cur_mat.albedo.spec;
        sum_clr.y += lights[i].clr.y * clr.y * diffuse_intens * cur_mat.albedo.diffus + 255.0 * specular_intens * cur_mat.albedo.spec;
        sum_clr.z += lights[i].clr.z * clr.z * diffuse_intens * cur_mat.albedo.diffus + 255.0 * specular_intens * cur_mat.albedo.spec;
    }

    clr = make_uchar4(min(ambient_intens * clr.x + sum_clr.x + cur_mat.albedo.refl * reflect_color.x + cur_mat.albedo.refr * refract_color.x, 255.0),
                      min(ambient_intens * clr.y + sum_clr.y + cur_mat.albedo.refl * reflect_color.y + cur_mat.albedo.refr * refract_color.y, 255.0),
                      min(ambient_intens * clr.z + sum_clr.z + cur_mat.albedo.refl * reflect_color.z + cur_mat.albedo.refr * refract_color.z, 255.0),
                      clr.w);
    return clr;
}

void ssaa(uchar4 *data, uchar4 *out, int w, int h, int sqrt_ray_num) {
    int w_scale = w * sqrt_ray_num;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int y_scale = i * sqrt_ray_num;
            int x_scale = j * sqrt_ray_num;
            uint4 aver_clr = make_uint4(0, 0, 0, 0);
            for (int n = y_scale; n < y_scale + sqrt_ray_num; n++) {
                for (int m = x_scale; m < x_scale + sqrt_ray_num; m++) {
                    aver_clr.x += data[n * w_scale + m].x;
                    aver_clr.y += data[n * w_scale + m].y;
                    aver_clr.z += data[n * w_scale + m].z;
                }
            }
            float d = sqrt_ray_num * sqrt_ray_num;
            out[i * w + j] = make_uchar4(aver_clr.x / d, aver_clr.y / d, aver_clr.z / d, aver_clr.w);
        }
    }
}

__global__ void kernel_ssaa(uchar4 *data, uchar4 *out, int w, int h, int sqrt_ray_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int w_scale = w * sqrt_ray_num;
    for (int i = idx; i < h; i += offsetx) {
        for (int j = idy; j < w; j += offsety) {
            int y_scale = i * sqrt_ray_num;
            int x_scale = j * sqrt_ray_num;
            uint4 aver_clr = make_uint4(0, 0, 0, 0);
            for (int n = y_scale; n < y_scale + sqrt_ray_num; n++) {
                for (int m = x_scale; m < x_scale + sqrt_ray_num; m++) {
                    aver_clr.x += data[n * w_scale + m].x;
                    aver_clr.y += data[n * w_scale + m].y;
                    aver_clr.z += data[n * w_scale + m].z;
                }
            }
            float d = sqrt_ray_num * sqrt_ray_num;
            out[i * w + j] = make_uchar4(aver_clr.x / d, aver_clr.y / d, aver_clr.z / d, aver_clr.w);
        }
    }
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int f_size, int f_txt_size, trig *trigs, Light *lights, int lights_num, int max_deep, int n) {
    float dw = 2.0 / (w - 1);
    float dh = 2.0 / (h - 1);
    float z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3 v = {-1.f + dw * i, (-1.f + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, dir, 0, floor, f_size, f_txt_size, trigs, lights, lights_num, max_deep, n);
        }
    }
}

__global__ void kernel_render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int f_size, int f_txt_size, trig *trigs, Light *lights, int l_num, int max_deep, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float dw = 2.0 / (w - 1);
    float dh = 2.0 / (h - 1);
    float z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = prod(bx, bz);
    for (int j = idy; j < h; j += offsety) {
        for (int i = idx; i < w; i += offsetx) {
            vec3 v = {-1.f + dw * i, (-1.f + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, dir, 0, floor, f_size, f_txt_size, trigs, lights, l_num, max_deep, n);
        }
    }
}

int main(int argc, char *argv[]) {
    bool on_gpu = true;
    string cmd = argv[1];

    int frame_num = 120;
    char path_to_image[100] = "res/%d.data";
    int w = 640, h = 480, angle = 120;
    float rc0 = 7.0, zc0 = 3.0, phic0 = 0.0, acr = 2.0, acz = 1.0, wcr = 2.0, wcz = 6.0, wcphi = 1.0, pcr = 0.0, pcz = 0.0;
    float rn0 = 2.0, zn0 = 0.0, phin0 = 0.0, anr = 0.5, anz = 0.1, wnr = 1.0, wnz = 4.0, wnphi = 1.0, pnr = 0.0, pnz = 0.0;

    float center1_x = 3.0, center1_y = -2.0, center1_z = 2.5, color1_x = 1.0, color1_y = 0.0, color1_z = 1.0, r1 = 1.5, refl_1 = 0.2, tran_1 = 0.8;
    float center2_x = -1.0, center2_y = -1.5, center2_z = 1.5, color2_x = 0.2, color2_y = 1.0, color2_z = 0.2, r2 = 1.5, refl_2 = 0.5, tran_2 = 0.0;
    float center3_x = -2.5, center3_y = 2.5, center3_z = 2.5, color3_x = 0.0, color3_y = 1.0, color3_z = 1.0, r3 = 1.5, refl_3 = 0.2, tran_3 = 0.9;
    int first_num = 0, second_num = 0, third_num = 0;

    float f1_x = -5.0, f1_y = -5.0, f1_z = 0.0, f2_x = -5.0, f2_y = 5.0, f2_z = 0.0, f3_x = 5.0, f3_y = 5.0, f3_z = 0.0, f4_x = 5.0, f4_y = -5.0, f4_z = 0.0;
    char path_to_floor[100] = "board.data";
    float f_col_r = 1.0, f_col_g = 1.0, f_col_b = 1.0, f_refl = 0.7;

    int lights_num = 1;
    float l_pos_x = 5.0, l_pos_y = 5.0, l_pos_z = 5.0;
    float light_r = 1.0, light_g = 1.0, light_b = 1.0, light_int = 1.6;

    int max_deep = 2, sqrt_ray_num = 4;

    if (argc > 2) {
        cerr << "Wrong input: too many args\n";
        exit(-1);
    }
    else if (argc == 2) {
        if (cmd == "--default") {
            cout << frame_num << '\n';
            cout << path_to_image << '\n';
            cout << w << ' ' << h << ' ' << angle << '\n';
            cout << rc0 << ' ' << zc0 << ' ' << phic0 << ' ' << acr << ' ' << acz << ' ' << wcr << ' ' << wcz << ' ' << wcphi << ' ' << pcr << ' ' << pcz << '\n';
            cout << rn0 << ' ' << zn0 << ' ' << phin0 << ' ' << anr << ' ' << anz << ' ' << wnr << ' ' << wnz << ' ' << wnphi << ' ' << pnr << ' ' << pnz << '\n';
            cout << center1_x << ' ' << center1_y << ' ' << center1_z << ' ' << color1_x << ' ' << color1_y << ' ' << color1_z << ' ' << r1 << ' ' << refl_1 << ' ' << tran_1 << ' ' << first_num << '\n';
            cout << center2_x << ' ' << center2_y << ' ' << center2_z << ' ' << color2_x << ' ' << color2_y << ' ' << color2_z << ' ' << r2 << ' ' << refl_2 << ' ' << tran_2 << ' ' << second_num << '\n';
            cout << center3_x << ' ' << center3_y << ' ' << center3_z << ' ' << color3_x << ' ' << color3_y << ' ' << color3_z << ' ' << r3 << ' ' << refl_3 << ' ' << tran_3 << ' ' << third_num << '\n';
            cout << f1_x << ' ' << f1_y << ' ' << f1_z << ' ' << f2_x << ' ' << f2_y << ' ' << f2_z << ' ' << f3_x << ' ' << f3_y << ' ' << f3_z << ' ' << f4_x << ' ' << f4_y << ' ' << f4_z << '\n';
            cout << path_to_floor << '\n';
            cout << f_col_r << ' ' << f_col_g << ' ' << f_col_b << ' ' << f_refl << '\n';
            cout << lights_num << '\n';
            cout << l_pos_x << ' ' << l_pos_y << ' ' << l_pos_z << '\n';
            cout << light_r << ' ' << light_g << ' ' << light_b << '\n';
            cout << max_deep << ' ' << sqrt_ray_num << '\n';
            return 0;
        }
        else if (cmd == "--cpu") {
            on_gpu = false;
        }
    }

    cin >> frame_num;
    cin >> path_to_image;
    cin >> w >> h >> angle;
    cin >> rc0 >> zc0 >> phic0 >> acr >> acz >> wcr >> wcz >> wcphi >> pcr >> pcz;
    cin >> rn0 >> zn0 >> phin0 >> anr >> anz >> wnr >> wnz >> wnphi >> pnr >> pnz;
    cin >> center1_x >> center1_y >> center1_z >> color1_x >> color1_y >> color1_z >> r1 >> refl_1 >> tran_1 >> first_num;
    cin >> center2_x >> center2_y >> center2_z >> color2_x >> color2_y >> color2_z >> r2 >> refl_2 >> tran_2 >> second_num;
    cin >> center3_x >> center3_y >> center3_z >> color3_x >> color3_y >> color3_z >> r3 >> refl_3 >> tran_3 >> third_num;
    cin >> f1_x >> f1_y >> f1_z >> f2_x >> f2_y >> f2_z >> f3_x >> f3_y >> f3_z >> f4_x >> f4_y >> f4_z;
    cin >> path_to_floor;
    cin >> f_col_r >> f_col_g >> f_col_b >> f_refl;
    cin >> lights_num;

    Light *lights = (Light *)malloc(sizeof(Light) * lights_num);
    for (int i = 0; i < lights_num; i++) {
        cin >> l_pos_x >> l_pos_y >> l_pos_z;
        cin >> light_r >> light_g >> light_b;
        lights[i] = {{l_pos_x, l_pos_y, l_pos_z}, light_int, {light_r, light_g, light_b}};
    }
    cin >> max_deep >> sqrt_ray_num;

    char buff[256];
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h * sqrt_ray_num * sqrt_ray_num);
    uchar4 *short_data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    vector<trig> figures;

    uchar4 *dev_data;
    uchar4 *dev_short_data;
    trig *dev_figures;
    Light *dev_lights;
    uchar4 *dev_floor;

    vec3 pc, pv;

    int w1, h1;
    FILE *fp = fopen(path_to_floor, "rb");
    fread(&w1, sizeof(int), 1, fp);
    fread(&h1, sizeof(int), 1, fp);
    uchar4 *floor = (uchar4 *)malloc(sizeof(uchar4) * w1 * h1);
    fread(floor, sizeof(uchar4), w1 * h1, fp);
    fclose(fp);
    floor_painting(floor, {f_col_r, f_col_g, f_col_b}, w1, h1);
    int f_size = (int)abs(f1_x - f3_x);

    trig fl1 = {{f1_x, f1_y, f1_z}, {f2_x, f2_y, f2_z}, {f3_x, f3_y, f3_z}};
    trig fl2 = {{f1_x, f1_y, f1_z}, {f3_x, f3_y, f3_z}, {f4_x, f4_y, f4_z}};
    material *materials = (material *)malloc(sizeof(material) * 4); 
    materials[0] = {1., 125.0, make_uchar4(color1_x * 255, color1_y * 255, color1_z * 255, 0), {0.2, 0.4, refl_1, tran_1}}; // purple
    materials[1] = {1., 125.0, make_uchar4(color2_x * 255, color2_y * 255, color2_z * 255, 0), {0.2, 0.3, refl_2, tran_2}}; // blue
    materials[2] = {1., 125.0, make_uchar4(color3_x * 255, color3_y * 255, color3_z * 255, 0), {0.7, 0.5, refl_3, tran_3}}; // green
    materials[3] = {1., 125.0, make_uchar4(f_col_r * 255, f_col_g * 255, f_col_b * 255, 0), {0.6, 0.7, f_refl, 0.0}};
    build_space(figures, {center1_x, center1_y, center1_z}, {center2_x, center2_y, center2_z}, {center3_x, center3_y, center3_z}, fl1, fl2, {r1, r2, r3}, materials);

    if (on_gpu) {
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h * sqrt_ray_num * sqrt_ray_num));
        CSC(cudaMalloc(&dev_short_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc((trig **)(&dev_figures), figures.size() * sizeof(trig)));
        CSC(cudaMalloc(&dev_lights, sizeof(Light) * lights_num));
        CSC(cudaMalloc(&dev_floor, sizeof(uchar4) * w1 * h1));

        CSC(cudaMemcpy(dev_figures, figures.data(), figures.size() * sizeof(trig), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_lights, lights, sizeof(Light) * lights_num, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(dev_floor, floor, sizeof(uchar4) * w1 * h1, cudaMemcpyHostToDevice));
    }

    for (int k = 0; k < frame_num; k++) {
        float step = 2 * M_PI * k / frame_num;

        float rct = rc0 + acr * sin(wcr * step + pcr);
        float zct = zc0 + acz * sin(wcz * step + pcz);
        float phict = phic0 + wcphi * step;

        float rnt = rn0 + anr * sin(wnr * step + pnr);
        float znt = zn0 + anz * sin(wnz * step + pnz);
        float phint = phin0 + wnphi * step;

        pc = {rct * cos(phict), rct * sin(phict), zct};
        pv = {rnt * cos(phint), rnt * sin(phint), znt};

        cudaEvent_t begin, end;
        float gpu_time = 0.0;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        if (on_gpu) {
            kernel_render<<<dim3(8, 32), dim3(8, 32)>>>(pc, pv, w * sqrt_ray_num, h * sqrt_ray_num, angle, dev_data, dev_floor, f_size, w1, dev_figures, dev_lights, lights_num, max_deep, figures.size());
            CSC(cudaGetLastError());

            kernel_ssaa<<<dim3(8, 32), dim3(8, 32)>>>(dev_data, dev_short_data, w, h, sqrt_ray_num);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(data, dev_short_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        }
        else {
            render(pc, pv, w * sqrt_ray_num, h * sqrt_ray_num, angle, data, floor, f_size, w1, figures.data(), lights, lights_num, max_deep, figures.size());
            ssaa(data, short_data, w, h, sqrt_ray_num);
            memcpy(data, short_data, sizeof(uchar4) * w * h);
        }

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpu_time, begin, end);

        cout << "Frame number: " << k << '\t' << "Time to calculate 1 frame: " << gpu_time << '\t' << "Number of rays " << w * h * sqrt_ray_num * sqrt_ray_num << '\n';
        sprintf(buff, path_to_image, k);
        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }

    free(data);
    free(short_data);
    free(floor);
    free(materials);

    if (on_gpu) {
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_short_data));
        CSC(cudaFree(dev_figures));
        CSC(cudaFree(dev_lights));
        CSC(cudaFree(dev_floor));
    }
    return 0;
}