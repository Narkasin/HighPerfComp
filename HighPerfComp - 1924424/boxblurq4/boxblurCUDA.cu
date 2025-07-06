#include <stdio.h>
#include <cuda_runtime.h>
#include "lodepng.h"
#include <png.h>

struct Pixel {
    unsigned char r, g, b, a;
};

bool loadPNG(const char* filename, Pixel** image, int* width, int* height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return false;

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) return false;

    png_infop info = png_create_info_struct(png);
    if (!info) return false;

    if (setjmp(png_jmpbuf(png))) return false;

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++)
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));

    png_read_image(png, row_pointers);

    *image = (Pixel*)malloc((*width) * (*height) * sizeof(Pixel));
    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            png_bytep px = &(row_pointers[y][x * 4]);
            (*image)[y * (*width) + x] = { px[0], px[1], px[2], px[3] };
        }
        free(row_pointers[y]);
    }
    free(row_pointers);
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    return true;
}

bool savePNG(const char* filename, Pixel* image, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return false;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) return false;

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);
    png_bytep row = (png_bytep)malloc(width * 4);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Pixel p = image[y * width + x];
            row[x * 4 + 0] = p.r;
            row[x * 4 + 1] = p.g;
            row[x * 4 + 2] = p.b;
            row[x * 4 + 3] = p.a;
        }
        png_write_row(png, row);
    }
    free(row);
    png_write_end(png, NULL);
    fclose(fp);
    png_destroy_write_struct(&png, &info);
    return true;
}
//Thomas Holloway - 1924424

__global__ void boxBlur(Pixel* input, Pixel* output, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    int x = i % w, y = i / w;
    int r = 0, g = 0, b = 0, n = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                Pixel p = input[ny * w + nx];
                r += p.r; g += p.g; b += p.b; n++;
            }
        }
    }
    output[i] = { (unsigned char)(r/n), (unsigned char)(g/n), (unsigned char)(b/n), input[i].a };
}

int main() {
    const char* inFile = "smaug.png";
    Pixel *h_input, *h_output;
    int w, h;

    if (!loadPNG(inFile, &h_input, &w, &h)) return 1;
    size_t size = w * h * sizeof(Pixel);

    Pixel *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (w * h + threads - 1) / threads;
    boxBlur<<<blocks, threads>>>(d_input, d_output, w, h);
    cudaDeviceSynchronize();

    h_output = (Pixel*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    char outFile[256];
    snprintf(outFile, sizeof(outFile), "smaug_blurred2.png");
    savePNG(outFile, h_output, w, h);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    return 0;
}
