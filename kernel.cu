#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <curand_kernel.h>

#define IS_ZERO(val) val > -EPSILON && val < EPSILON

#define WIDTH 2000
#define HEIGHT 2000
#define FRAME_COUNT 10
#define IS_ANIMATION false

#define EPSILON 0.0001
#define M_PI 3.14159265

#define CROSS(v1, v2) vec3((v1).y * (v2).z - (v1).z * (v2).y, (v1).z * (v2).x - (v1).x * (v2).z, (v1).x * (v2).y - (v1).y * (v2).x)
#define DOT(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)
#define DOT_2(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y)
#define LENGTH(v) (sqrt(DOT(v, v)))
#define LENGTH_2(v) (sqrt(DOT_2(v, v)))
#define NORMALIZE(v) ((v) / LENGTH(v))

using namespace std;

class ivec2 {
public:
    int x;
    int y;

    ivec2() : x(0), y(0) {}
    ivec2(int x, int y) : x(x), y(y) {}
};

struct vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ vec3() : x(0.0), y(0.0), z(0.0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3(const vec3& other) : x(other.x), y(other.y), z(other.z) {}

    __host__ __device__ vec3 operator+(const vec3& other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }
    __host__ __device__ vec3 operator-(const vec3& other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }
    __host__ __device__ vec3 operator*(const vec3& other) const {
        return vec3(x * other.x, y * other.y, z * other.z);
    }
    __host__ __device__ vec3 operator/(const vec3& other) const {
        return vec3(x / other.x, y / other.y, z / other.z);
    }
    __host__ __device__ vec3 operator*(float scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }
    __host__ __device__ vec3 operator/(float scalar) const {
        return vec3(x / scalar, y / scalar, z / scalar);
    }
};

struct vec2 {
    float x;
    float y;

    __device__ vec2() : x(0.0), y(0.0) {}
    __device__ vec2(float x, float y) : x(x), y(y) {}
    __device__ vec2(const vec2& other) : x(other.x), y(other.y) {}

    __device__ vec2 operator+(const vec2& other) const {
        return vec2(x + other.x, y + other.y);
    }
    __device__ vec2 operator-(const vec2& other) const {
        return vec2(x - other.x, y - other.y);
    }
    __device__ vec2 operator*(const vec2& other) const {
        return vec2(x * other.x, y * other.y);
    }
    __device__ vec2 operator/(const vec2& other) const {
        return vec2(x / other.x, y / other.y);
    }
    __device__ vec2 operator*(float scalar) const {
        return vec2(x * scalar, y * scalar);
    }
    __device__ vec2 operator/(float scalar) const {
        return vec2(x / scalar, y / scalar);
    }
};

class fcolor {
public:
    float red;
    float green;
    float blue;

    __host__ __device__ fcolor() : red(0.0), green(0.0), blue(0.0) {}
    __host__ __device__ fcolor(float red, float green, float blue) : red(red), green(green), blue(blue) {}
    __host__ __device__ fcolor(vec3 vector_color) : red(vector_color.x), green(vector_color.y), blue(vector_color.z) {}

    __host__ __device__ fcolor operator+(const fcolor& other) const {
        return fcolor(red + other.red, green + other.green, blue + other.blue);
    }
    __host__ __device__ fcolor operator-(const fcolor& other) const {
        return fcolor(red - other.red, green - other.green, blue - other.blue);
    }
    __host__ __device__ fcolor operator*(const fcolor& other) const {
        return fcolor(red * other.red, green * other.green, blue * other.blue);
    }
    __host__ __device__ fcolor operator*(float scalar) const {
        return fcolor(red * scalar, green * scalar, blue * scalar);
    }
    __host__ __device__ fcolor operator/(float scalar) const {
        return fcolor(red / scalar, green / scalar, blue / scalar);
    }
};

class icolor {
public:
    int red;
    int green;
    int blue;

    __host__ __device__ icolor() : red(0), green(0), blue(0) {}
    __host__ __device__ icolor(int red, int green, int blue) : red(red), green(green), blue(blue) {}
    __host__ __device__ icolor(fcolor other) : red((int)floor(other.red * 255.0)), green((int)floor(other.green * 255.0)), blue((int)floor(other.blue * 255.0)) {}

    __host__ __device__ icolor operator+(const icolor& other) const {
        return icolor(red + other.red, green + other.green, blue + other.blue);
    }

    __host__ __device__ icolor operator-(const icolor& other) const {
        return icolor(red - other.red, green - other.green, blue - other.blue);
    }

    __host__ __device__ icolor operator*(float scalar) const {
        return icolor((int)floor(red * scalar), (int)floor(green * scalar), (int)floor(blue * scalar));
    }

    __host__ __device__ icolor operator/(float scalar) const {
        return icolor((int)floor(red / scalar), (int)floor(green / scalar), (int)floor(blue / scalar));
    }

    __host__ __device__ icolor copy() const {
        return icolor(red, green, blue);
    }
};

__device__ curandState_t get_rand_state(unsigned int seed) {
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    return state;
}

__device__ float get_random_float(curandState_t& state) {
    return curand_uniform(&state);
}

__global__ void calculate_pixel(ivec2* positions, icolor* colors, int size, int width, int height, int index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int x = positions[tid].x;
        int y = positions[tid].y;

        vec2 uv = vec2(x, y) / vec2(width, height);

        fcolor out_color = fcolor(uv.x, uv.y, 0.0);

        // OUTPUT
        colors[tid] = icolor(out_color);
    }
}

class GPU_Image {
public:
    string name;
    int width;
    int height;
    int i;
    unsigned char* imageData;

    GPU_Image(const string& name, int width, int height, int i) : name(name), width(width), height(height), i(i) {}

    void save_bitmap(const string& filename)
    {
        // Define the bitmap file header
        unsigned char bitmapFileHeader[14] = {
                'B', 'M',                     // Signature
                0, 0, 0, 0,           // File size (to be filled later)
                0, 0, 0, 0,           // Reserved
                54, 0, 0, 0        // Pixel data offset
        };

        // Define the bitmap info header
        unsigned char bitmapInfoHeader[40] = {
                40, 0, 0, 0,            // Info header size
                0, 0, 0, 0,             // Image width (to be filled later)
                0, 0, 0, 0,           // Image height (to be filled later)
                1, 0,                         // Number of color planes
                24, 0,                        // Bits per pixel (24 bits for RGB)
                0, 0, 0, 0,          // Compression method (none)
                0, 0, 0, 0,          // Image size (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Horizontal resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Vertical resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Number of colors in the palette (not used for 24-bit images)
                0, 0, 0, 0           // Number of important colors (not used for 24-bit images)
        };

        // Calculate the padding bytes
        int paddingSize = (4 - (width * 3) % 4) % 4;

        // Calculate the file size
        int fileSize = 54 + (width * height * 3) + (paddingSize * height);

        // Fill in the file size in the bitmap file header
        bitmapFileHeader[2] = (unsigned char)(fileSize);
        bitmapFileHeader[3] = (unsigned char)(fileSize >> 8);
        bitmapFileHeader[4] = (unsigned char)(fileSize >> 16);
        bitmapFileHeader[5] = (unsigned char)(fileSize >> 24);

        // Fill in the image width in the bitmap info header
        bitmapInfoHeader[4] = (unsigned char)(width);
        bitmapInfoHeader[5] = (unsigned char)(width >> 8);
        bitmapInfoHeader[6] = (unsigned char)(width >> 16);
        bitmapInfoHeader[7] = (unsigned char)(width >> 24);

        // Fill in the image height in the bitmap info header
        bitmapInfoHeader[8] = (unsigned char)(height);
        bitmapInfoHeader[9] = (unsigned char)(height >> 8);
        bitmapInfoHeader[10] = (unsigned char)(height >> 16);
        bitmapInfoHeader[11] = (unsigned char)(height >> 24);

        // Open the output file
        ofstream file(filename, ios::binary);

        // Write the bitmap headers
        file.write(reinterpret_cast<const char*>(bitmapFileHeader), sizeof(bitmapFileHeader));
        file.write(reinterpret_cast<const char*>(bitmapInfoHeader), sizeof(bitmapInfoHeader));

        // Write the pixel data (BGR format) row by row
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                // Calculate the pixel position
                int position = (x + y * width) * 3;

                // Write the pixel data (BGR order)
                file.write(reinterpret_cast<const char*>(&imageData[position + 2]), 1); // Blue
                file.write(reinterpret_cast<const char*>(&imageData[position + 1]), 1); // Green
                file.write(reinterpret_cast<const char*>(&imageData[position]), 1);     // Red
            }

            // Write the padding bytes
            for (int i = 0; i < paddingSize; i++)
            {
                file.write("\0", 1);
            }
        }

        // Close the file
        file.close();
    }

    void put_pixel(const int x, const int y, const int r, const int g, const int b) {
        int position = (x + y * width) * 3;
        imageData[position] = r;
        imageData[position + 1] = g;
        imageData[position + 2] = b;
    }

    void generate() {
        imageData = new unsigned char[width * height * 3];

        const int size = width * height;

        // DEFINE PARAMETERS

        ivec2* positions = new ivec2[size];
        icolor* colors = new icolor[size];

        // DEFINE SIZES

        const long vector2int_size = size * sizeof(ivec2);
        const long colorint_size = size * sizeof(icolor);

        // UPDATE I/O PARAMERTERS

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            positions[i] = ivec2(x, y);
            colors[i] = icolor();
        }

        // DEFINE D_PARAMETERS

        ivec2* d_positions;
        icolor* d_colors;

        cudaMalloc((void**)&d_positions, vector2int_size);
        cudaMalloc((void**)&d_colors, colorint_size);

        // MEMORY COPY PARAMETERS

        cudaMemcpy(d_positions, positions, vector2int_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, colors, colorint_size, cudaMemcpyHostToDevice);

        // RUN

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        calculate_pixel << < blocksPerGrid, threadsPerBlock >> > (d_positions, d_colors, size, width, height, i);

        cudaMemcpy(colors, d_colors, colorint_size, cudaMemcpyDeviceToHost);

        // PROCESS OUTPUT

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            icolor color = colors[i];
            put_pixel(x, height - y - 1, color.red, color.green, color.blue);
        }

        save_bitmap("./output/" + name + ".bmp");

        // FREE MEMORY

        cudaFree(d_positions);
        cudaFree(d_colors);

        delete[] positions;
        delete[] colors;

        delete[] imageData;
    }
};

vector<string> split_string(const string& str, const string& delimiter) {
    vector<string> tokens;
    size_t start = 0;
    size_t end = 0;
    while ((end = str.find(delimiter, start)) != string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

string sec_to_time(float time) {
    float n_time = time;
    string suffix = " second(s)";
    if (n_time > 60.0f * 60.0f * 24.0f) {
        n_time /= 60.0f * 60.0f * 24.0f;
        suffix = " day(s)";
    }
    else if (n_time > 60.0f * 60.0f) {
        n_time /= 60.0f * 60.0f;
        suffix = " hour(s)";
    }
    else if (n_time > 60.0f) {
        n_time /= 60.0f;
        suffix = " minute(s)";
    }
    return to_string(n_time) + suffix;
}

int main() {
    chrono::system_clock::time_point start_time = chrono::system_clock::now();

    if (IS_ANIMATION) {
        for (int i = 0; i < FRAME_COUNT; i++) {
            chrono::system_clock::time_point start_frame = chrono::system_clock::now();

            string name = "out_" + to_string(i);

            GPU_Image main = GPU_Image(name, WIDTH, HEIGHT, i);
            main.generate();

            chrono::time_point<chrono::system_clock> end_frame = chrono::system_clock::now();
            chrono::duration<float> duration_frame = end_frame - start_frame;

            cout << "RENDERED: " << i + 1 << "/" << FRAME_COUNT << " (" << floor((float)(i + 1.0) / (float)FRAME_COUNT * 1000.0) / 10.0 << "%)" << " " << sec_to_time(duration_frame.count()) << " | ETA: " << sec_to_time((float)(FRAME_COUNT - (i + 1)) * duration_frame.count()) << endl;
        }
    }
    else {
        GPU_Image main = GPU_Image("out_image", WIDTH, HEIGHT, 0);
        main.generate();
    }

    chrono::time_point<chrono::system_clock> end_time = chrono::system_clock::now();
    chrono::duration<float> duration = end_time - start_time;
    cout << "Total time taken: " << sec_to_time(duration.count()) << endl;

    return 0;
}
