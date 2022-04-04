#include <iostream>
#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <vector>

using namespace std;

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

struct vec3 {
	float x;
	float y;
	float z;
};

struct trig {
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
};

__host__ __device__ float dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(vec3 a, vec3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ vec3 norm(vec3 v) {
	float l = sqrt(dot(v, v));
	return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	return {a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z};
}

__host__ __device__ vec3 const_mult(vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__  double vector_length(vec3 a) {
    return sqrt(dot(a, a));
}

void print(vec3 v) {
	printf("%e %e %e\n", v.x, v.y, v.z);
}

vec3 get_color(vec3 color) {
    return {color.x * (float)255., color.y * (float)255., color.z * (float)255.};
}

__host__ __device__  uchar4 to_uchar4 (vec3 b) {
    return make_uchar4(b.x, b.y, b.z, 0);
}

void floor_painting(uchar4 *floor, vec3 color, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        floor[i] = make_uchar4(color.x * floor[i].x, color.y * floor[i].y, color.z * floor[i].z, floor[i].w);
    }
}

void build_space(vector<trig> &figures, const vec3 &center1, const vec3 &center2, const vec3 &center3, const vec3 &color1, const vec3 &color2, const vec3 &color3, const vec3 &color4, const trig &fl1, const trig &fl2, const vec3 &rads) {
	//Tetrahedron
    vec3 color = get_color(color1);

    vec3 point_a = {rads.x + center1.x, rads.x + center1.y, rads.x + center1.z};
    vec3 point_b = {rads.x + center1.x, -rads.x + center1.y, -rads.x + center1.z};
    vec3 point_c = {-rads.x + center1.x, rads.x + center1.y, -rads.x + center1.z};
    vec3 point_d = {-rads.x + center1.x, -rads.x + center1.y, rads.x + center1.z};

    figures.push_back({point_a, point_b, point_d, to_uchar4(color)});
    figures.push_back({point_a, point_c, point_d, to_uchar4(color)});
    figures.push_back({point_b, point_c, point_d, to_uchar4(color)});
    figures.push_back({point_a, point_b, point_c, to_uchar4(color)});

    //Octahedron
    color = get_color(color2);

    point_a = {center2.x, center2.y - rads.y, center2.z};
    point_b = {center2.x - rads.y, center2.y, center2.z};
    point_c = {center2.x, center2.y, center2.z - rads.y};
    point_d = {center2.x + rads.y, center2.y, center2.z};
    vec3 point_e = {center2.x, center2.y, center2.z + rads.y};
    vec3 point_f = {center2.x, center2.y + rads.y, center2.z};

    figures.push_back({point_a, point_b, point_c, to_uchar4(color)});
    figures.push_back({point_a, point_c, point_d, to_uchar4(color)});
    figures.push_back({point_a, point_d, point_e, to_uchar4(color)});
    figures.push_back({point_a, point_e, point_b, to_uchar4(color)});
    figures.push_back({point_f, point_c, point_b, to_uchar4(color)});
    figures.push_back({point_f, point_d, point_c, to_uchar4(color)});
    figures.push_back({point_f, point_e, point_d, to_uchar4(color)});
    figures.push_back({point_f, point_b, point_e, to_uchar4(color)});

    //Dodecahedron
    color = get_color(color3);

    float phi = (1.0 + sqrt(5.0)) / 2.0;
    point_a = {center3.x + (-1.f / phi / sqrt(3.f) * rads.z), center3.y, center3.z + (phi / sqrt(3.f) * rads.z)};
    point_b = {center3.x + (1.f / phi / sqrt(3.f) * rads.z), center3.y, center3.z + (phi / sqrt(3.f) * rads.z)};
    point_c = {center3.x + (-1.f / sqrt(3.f) * rads.z), center3.y + (1.f / sqrt(3.f) * rads.z), center3.z + (1.f / sqrt(3.f) * rads.z)};
    point_d = {center3.x + (1.f / sqrt(3.f) * rads.z), center3.y + (1.f / sqrt(3.f) * rads.z), center3.z + (1.f / sqrt(3.f) * rads.z)};
    point_e = {center3.x + (1.f / sqrt(3.f) * rads.z), center3.y + (-1.f / sqrt(3.f) * rads.z), center3.z + (1.f / sqrt(3.f) * rads.z)};
    point_f = {center3.x + (-1.f / sqrt(3.f) * rads.z), center3.y + (-1.f / sqrt(3.f) * rads.z), center3.z + (1.f / sqrt(3.f) * rads.z)};
    vec3 point_g = {center3.x, center3.y + (-phi / sqrt(3.f) * rads.z), center3.z + (1.f / phi / sqrt(3.f) * rads.z)};
    vec3 point_h = {center3.x, center3.y + (phi / sqrt(3.f) * rads.z), center3.z + (1.f / phi / sqrt(3.f) * rads.z)};
    vec3 point_i = {center3.x + (-phi / sqrt(3.f) * rads.z), center3.y + (-1.f / phi / sqrt(3.f) * rads.z), center3.z};
    vec3 point_j = {center3.x + (-phi / sqrt(3.f) * rads.z), center3.y + (1.f / phi / sqrt(3.f) * rads.z), center3.z};
    vec3 point_k = {center3.x + (phi / sqrt(3.f) * rads.z), center3.y + (1.f / phi / sqrt(3.f) * rads.z), center3.z};
    vec3 point_l = {center3.x + (phi / sqrt(3.f) * rads.z), center3.y + (-1.f / phi / sqrt(3.f) * rads.z), center3.z};
    vec3 point_m = {center3.x, center3.y + (-phi / sqrt(3.f) * rads.z), center3.z + (-1.f / phi / sqrt(3.f) * rads.z)};
    vec3 point_n = {center3.x, center3.y + (phi / sqrt(3.f) * rads.z), center3.z + (-1.f / phi / sqrt(3.f) * rads.z)};
    vec3 point_o = {center3.x + (1.f / sqrt(3.f) * rads.z), center3.y + (1.f / sqrt(3.f) * rads.z), center3.z + (-1.f / sqrt(3.f) * rads.z)};
    vec3 point_p = {center3.x + (1.f / sqrt(3.f) * rads.z), center3.y + (-1.f / sqrt(3.f) * rads.z), center3.z + (-1.f / sqrt(3.f) * rads.z)};
    vec3 point_q = {center3.x + (-1.f / sqrt(3.f) * rads.z), center3.y + (-1.f / sqrt(3.f) * rads.z), center3.z + (-1.f / sqrt(3.f) * rads.z)};
    vec3 point_r = {center3.x + (-1.f / sqrt(3.f) * rads.z), center3.y + (1.f / sqrt(3.f) * rads.z), center3.z + (-1.f / sqrt(3.f) * rads.z)};
    vec3 point_s = {center3.x + (1.f / phi / sqrt(3.f) * rads.z), center3.y, center3.z + (-phi / sqrt(3.f) * rads.z)};
    vec3 point_t = {center3.x + (-1.f / phi / sqrt(3.f) * rads.z), center3.y, center3.z + (-phi / sqrt(3.f) * rads.z)};

    figures.push_back({point_e, point_a, point_g, to_uchar4(color)});
    figures.push_back({point_a, point_f, point_g, to_uchar4(color)});
    figures.push_back({point_e, point_b, point_a, to_uchar4(color)});
    figures.push_back({point_h, point_a, point_d, to_uchar4(color)});
    figures.push_back({point_a, point_b, point_d, to_uchar4(color)});
    figures.push_back({point_h, point_c, point_a, to_uchar4(color)});
    figures.push_back({point_k, point_b, point_l, to_uchar4(color)});
    figures.push_back({point_b, point_e, point_l, to_uchar4(color)});
    figures.push_back({point_k, point_d, point_b, to_uchar4(color)});
    figures.push_back({point_i, point_a, point_j, to_uchar4(color)});
    figures.push_back({point_a, point_c, point_j, to_uchar4(color)});
    figures.push_back({point_i, point_f, point_a, to_uchar4(color)});
    figures.push_back({point_m, point_f, point_q, to_uchar4(color)});
    figures.push_back({point_f, point_i, point_q, to_uchar4(color)});
    figures.push_back({point_m, point_g, point_f, to_uchar4(color)});
    figures.push_back({point_p, point_e, point_m, to_uchar4(color)});
    figures.push_back({point_e, point_g, point_m, to_uchar4(color)});
    figures.push_back({point_p, point_l, point_e, to_uchar4(color)});
    figures.push_back({point_r, point_c, point_n, to_uchar4(color)});
    figures.push_back({point_c, point_h, point_n, to_uchar4(color)});
    figures.push_back({point_r, point_j, point_c, to_uchar4(color)});
    figures.push_back({point_n, point_d, point_o, to_uchar4(color)});
    figures.push_back({point_d, point_k, point_o, to_uchar4(color)});
    figures.push_back({point_n, point_h, point_d, to_uchar4(color)});
    figures.push_back({point_t, point_i, point_r, to_uchar4(color)});
    figures.push_back({point_i, point_j, point_r, to_uchar4(color)});
    figures.push_back({point_t, point_q, point_i, to_uchar4(color)});
    figures.push_back({point_o, point_l, point_s, to_uchar4(color)});
    figures.push_back({point_l, point_p, point_s, to_uchar4(color)});
    figures.push_back({point_o, point_k, point_l, to_uchar4(color)});
    figures.push_back({point_s, point_m, point_t, to_uchar4(color)});
    figures.push_back({point_m, point_q, point_t, to_uchar4(color)});
    figures.push_back({point_s, point_p, point_m, to_uchar4(color)});
    figures.push_back({point_t, point_n, point_s, to_uchar4(color)});
    figures.push_back({point_n, point_o, point_s, to_uchar4(color)});
    figures.push_back({point_t, point_r, point_n, to_uchar4(color)});

    //Scene
    color = get_color(color4);

    figures.push_back({fl1.a, fl1.b, fl1.c, to_uchar4(color)});
    figures.push_back({fl2.a, fl2.b, fl2.c, to_uchar4(color)});
}

__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, trig* trigs, int n, vec3 light_source, vec3 light_shade) {
	int k, k_min = -1;
	float ts_min;
	for(k = 0; k < n; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		float u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		float v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		float ts = dot(q, e2) / div; 	
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if (k_min == -1)
		return {0, 0, 0, 0};
    vec3 pos_tmp = add(const_mult(dir, ts_min), pos);
    vec3 new_direction = norm(diff(light_source, pos_tmp));
    for (int i = 0; i < n; i++) {
        vec3 e1 = diff(trigs[i].b, trigs[i].a);
        vec3 e2 = diff(trigs[i].c, trigs[i].a);
        vec3 p = prod(new_direction, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        
        vec3 t = diff(pos_tmp, trigs[i].a);
        double u = dot(p, t) / div;
        
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, new_direction) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div;
        if (ts > 0.0 && ts < vector_length(diff(light_source, pos_tmp)) && i != k_min) {
            return {0, 0, 0, 0};
        }
    }
    uchar4 color_min = {0, 0, 0, 0};
    uchar4 result = trigs[k_min].color;

    color_min.x += result.x * light_shade.x;
    color_min.y += result.y * light_shade.y;
    color_min.z += result.z * light_shade.z;    
    color_min.w = 0;

    return color_min;
}

void render(vec3 pc, vec3 pv, int w, int h, float angle, uchar4 *data, trig* trigs, int n, vec3 light_source, vec3 light_shade) {
	int i, j;
	float dw = 2.0 / (w - 1.0);
	float dh = 2.0 / (h - 1.0);
	float z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
	vec3 by = norm(prod(bx, bz));
	for(i = 0; i < w; i++) {
		for(j = 0; j < h; j++) {
			vec3 v = {(float)-1.0 + dw * i, (float)(-1.0 + dh * j) * h / w, z};
			vec3 dir = mult(bx, by, bz, v);
			data[(h - 1 - j) * w + i] = ray(pc, norm(dir), trigs, n, light_source, light_shade);
		}
    }
}

__global__ void kernel_render(vec3 pc, vec3 pv, int w, int h, float angle, uchar4 *data, trig* trigs, int n, vec3 light_source, vec3 light_shade) {
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
    for (int j = idy; j < h; j+= offsety) {
        for (int i = idx; i < w; i+= offsetx) {
            vec3 v = {(float)-1.0 + dw * i, (float)(-1.0 + dh * j) * h / w, z};
            vec3 dir = norm(mult(bx, by, bz, v));
            data[(h - 1 - j) * w + i] = ray(pc, norm(dir), trigs, n, light_source, light_shade);
        }
    }
}

void ssaa(uchar4* data, uchar4* out, int w, int h, int sqrt_ray_num) {
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
            out[i*w + j] = make_uchar4(aver_clr.x / d, aver_clr.y / d, aver_clr.z / d, aver_clr.w);
        }
    }
}

__global__ void kernel_ssaa(uchar4* data, uchar4* out, int w, int h, int sqrt_ray_num) {
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

int main(int argc, char* argv[]) {
    bool on_gpu = true;
    string cmd = argv[1];

	int frame_num = 120;
	char path_to_image[100] = "res/%d.data";
	int w = 640, h = 480, angle = 120;
	float rc0 = 4.0, zc0 = 4.0, phic0 = 4.0, acr = 1.0, acz = 1.0, wcr = 1.0, wcz = 2.0, wcphi = 1.0, pcr = 0.0, pcz = 0.0;
    float rn0 = 0.5, zn0 = 0.5, phin0 = 0.1, anr = 1.0, anz = 0.5, wnr = 1.0, wnz = 0.5, wnphi = 1.0, pnr = 0.0, pnz = 0.0;
	float center1_x = 3.0, center1_y = -2.0, center1_z = 2.5, color1_x = 1.0, color1_y = 0.0, color1_z = 1.0, r1 = 1.5, refl_1 = 0.2, tran_1 = 0.8, first_num = 0.0;
    float center2_x = -1.0, center2_y = -1.5, center2_z = 1.5, color2_x = 0.2, color2_y = 1.0, color2_z = 0.2, r2 = 1.5, refl_2 = 0.5, tran_2 = 0.0, second_num = 0.0;
    float center3_x = -2.5, center3_y = 2.5, center3_z = 2.5, color3_x = 0.0, color3_y = 1.0, color3_z = 1.0, r3 = 1.5, refl_3 = 0.2, tran_3 = 0.9, third_num = 0.0;
	float f1_x = -5.0, f1_y = -5.0, f1_z = 0.0, f2_x = -5.0, f2_y = 5.0, f2_z = 0.0, f3_x = 5.0, f3_y = 5.0, f3_z= 0.0, f4_x = 5.0, f4_y = -5.0, f4_z = 0.0;
	char path_to_floor[100] = "board.data";
	float f_color_r = 1.0, f_color_g = 1.0, f_color_b = 1.0, f_ref = 0.7; 
	int lights_num = 1;
	float l_pos_x = 5.0, l_pos_y = 5.0, l_pos_z = 5.0;
    float light_r = 1.0, light_g = 1.0, light_b = 1.0;
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
            cout << f_color_r << ' ' << f_color_g << ' ' << f_color_b << ' ' << f_ref << '\n';
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
    cin >> f_color_r >> f_color_g >> f_color_b >> f_ref;
    cin >> lights_num;

    for (int i = 0; i < lights_num; i++) {
        cin >> l_pos_x >> l_pos_y >> l_pos_z;
        cin >> light_r >> light_g >> light_b;
    }

    cin >> max_deep >> sqrt_ray_num;
    
	char buff[256];
	int w_smooth = w * sqrt_ray_num;
	int h_smooth = h * sqrt_ray_num;
    int rays_count = w_smooth * h_smooth;

	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w_smooth * h_smooth);
	uchar4 *short_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	vec3 pc, pv;
    vec3 light_source = {l_pos_x, l_pos_y, l_pos_z};
    vec3 light_shade = {light_r, light_g, light_b};
	vector<trig> figures;

    uchar4 *dev_data;
    uchar4 *dev_short_data;
    trig* dev_figures;

	int w1, h1;
    FILE *fp = fopen(path_to_floor, "rb");
    fread(&w1, sizeof(int), 1, fp);
    fread(&h1, sizeof(int), 1, fp);
    uchar4 *floor = (uchar4 *)malloc(sizeof(uchar4) * w1 * h1);
    fread(floor, sizeof(uchar4), w1 * h1, fp);
    fclose(fp);
    floor_painting(floor, {f_color_r, f_color_g, f_color_b}, w1, h1);

	trig fl1 = {{f1_x, f1_y, f1_z}, {f2_x, f2_y, f2_z}, {f3_x, f3_y, f3_z}};
    trig fl2 = {{f1_x, f1_y, f1_z}, {f3_x, f3_y, f3_z}, {f4_x, f4_y, f4_z}};
    
    build_space(figures, {center1_x, center1_y, center1_z}, {center2_x, center2_y, center2_z},
                                  {center3_x, center3_y, center3_z}, {color1_x, color1_y, color1_z},
                                  {color2_x, color2_y, color2_z}, {color3_x, color3_y, color3_z},
                                  {f_color_r, f_color_g, f_color_b},
                                  fl1, fl2, {r1, r2, r3});
    
    if (on_gpu) {
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w_smooth * h_smooth));
        CSC(cudaMalloc(&dev_short_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc((trig **)(&dev_figures), figures.size() * sizeof(trig)));

        CSC(cudaMemcpy(dev_figures, figures.data(), figures.size() * sizeof(trig), cudaMemcpyHostToDevice));
    }

    float all_time = 0.0;
    float avg_time = 0.0;

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
		
        cudaEvent_t start, end;
        float time = 0.0;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&end));
        CSC(cudaEventRecord(start, 0));
        if (on_gpu) {
            kernel_render<<<dim3(8, 32), dim3(8, 32)>>>(pc, pv, w_smooth, h_smooth, angle, dev_data, dev_figures, figures.size(), light_source, light_shade);
            CSC(cudaGetLastError());

            kernel_ssaa<<<dim3(8, 32), dim3(8, 32)>>>(dev_data, dev_short_data, w, h, sqrt_ray_num);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(data, dev_short_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } else {
            render(pc, pv, w_smooth, h_smooth, angle, data, figures.data(), figures.size(), light_source, light_shade);
            ssaa(data, short_data, w, h, sqrt_ray_num);
            memcpy(data, short_data, sizeof(uchar4) * w * h);
        }

        CSC(cudaEventRecord(end, 0));
        CSC(cudaEventSynchronize(end));
        CSC(cudaEventElapsedTime(&time, start, end));
        CSC(cudaEventDestroy(start));
		CSC(cudaEventDestroy(end));

        all_time += time;
        cout << "Frame number: " << k << '\t' << "Time to calculate 1 frame: " << time << '\t' << "Number of rays " << rays_count << '\n';
		sprintf(buff, path_to_image, k);	

		FILE *out = fopen(buff, "wb");
		fwrite(&w, sizeof(int), 1, out);
		fwrite(&h, sizeof(int), 1, out);	
		fwrite(data, sizeof(uchar4), w * h, out);
		fclose(out);
	}
    avg_time = all_time / frame_num;
    cout << "All time to render all frames: " << "\t" << all_time << "Average time to render 1 frame: " << avg_time;
	free(data);
    free(short_data);
    free(floor);
    if (on_gpu) {
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_short_data));
        CSC(cudaFree(dev_figures));
    }
	return 0;
}