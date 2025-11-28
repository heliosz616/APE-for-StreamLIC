#include "ape.hpp"
#include "common.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm> // For std::copy
#include "top.hpp"

// ==================================================================================
// Kernel Configuration
// ==================================================================================
#define W_BITW 12
#define A_BITW 12
#define BF_BITW 40
#define PCN 12
#define SCM 16
#define BN1 27
#define BN2 9
#define BN3 12
#define MAX_W 80 // W dimension
#define H 48     // H dimension (fixed for this test)

// Set Cosim height to 4 for fast cosimulation. Reset it to 48 to simulate an entire frame
const int H_cosim=4;

// Calculated Dimensions
constexpr int W = MAX_W;
constexpr int C_y = SCM * PCN;         // Latent Y Channels
constexpr int C_gp = 2 * SCM * PCN;    // Global Params Channels
constexpr long long TOTAL_Y_ELEMENTS = (long long)H * W * C_y;
constexpr long long TOTAL_GP_ELEMENTS = (long long)H * W * C_gp;

// ==================================================================================
// Global Array Definitions (H, W, C)
// ==================================================================================

// Input Data Arrays
int Y_DATA[H][W][C_y];
int GP_DATA[H][W][C_gp];

//Golden ouput arrays;
int GOLDEN_SYMBOLS[H][W][C_y];
int GOLDEN_PDF_INDICES[H][W][C_y];

// Output Data Arrays
int OUTPUT_SYMBOLS[H][W][C_y];
int OUTPUT_PDF_INDICES[H][W][C_y];

// ==================================================================================
// Helper Functions
// ==================================================================================

// Generic file reader and populator for 3D arrays
template<typename T, int H_dim, int W_dim, int C_dim>
void load_file_to_3d_array(const std::string& filename, T (&arr)[H_dim][W_dim][C_dim], long long expected_elements) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        exit(1);
    }

    // Read raw data into a temporary flat vector
    std::vector<T> flat_vec;
    T val;
    while (file >> val) {
        flat_vec.push_back(val);
    }
    file.close();

    if (flat_vec.size() != expected_elements) {
        std::cerr << "Error: " << filename << " size mismatch. Expected " << expected_elements
                  << " but found " << flat_vec.size() << std::endl;
        exit(1);
    }

    // Copy flat data into the 3D array (H, W, C)
    // The Python generator produces (H*W*C) flattened in C-order (H-W-C)
    T* arr_ptr = (T*)arr;
    std::copy(flat_vec.begin(), flat_vec.end(), arr_ptr);
}

// Helper to load parameter vectors (can still use vector for parameters as they are small)
template<typename T>
void load_param_file_to_vector(const std::string& filename, std::vector<T>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open parameter file " << filename << std::endl;
        exit(1);
    }
    T val;
    while (file >> val) {
        vec.push_back(val);
    }
    file.close();
}


// ==================================================================================
// Main Testbench
// ==================================================================================

int main() {
    // 1. Instantiate the DUT
    // ------------------------------------------------------------------------------
//    APEngine<W_BITW, A_BITW, BF_BITW, PCN, SCM, BN1, BN2, BN3, MAX_W> dut;
//    dut.init();

    // 2. Define Streams
    // ------------------------------------------------------------------------------
    hls::stream<ap_int<W_BITW>> weight_stream;
    hls::stream<ap_uint<10>> spatial_offset_stream;
    hls::stream<ap_uint<10>> channel_index_stream;
    hls::stream<scale_T> scale_param_stream;
    hls::stream<bias_T> bias_stream;

    hls::stream<AggData<ap_int<A_BITW>, PCN>> latent_stream;
    hls::stream<AggData<ap_int<A_BITW>, PCN>> add_stream1;
    hls::stream<AggData<ap_int<A_BITW>, PCN>> add_stream2;
    hls::stream<AggData<ap_int<8>, PCN>> symbol_stream;
    hls::stream<AggData<ap_int<8>, PCN>> pdf_index_stream;

    // 3. Load Parameters from Files (Using vectors for small parameter files)
    // ------------------------------------------------------------------------------
    std::cout << "[TB] Loading parameter files..." << std::endl;
    std::vector<int> weights_raw, ch_indices_raw, offsets_raw, biases_raw, sc_params_raw, sc_shifts_raw;

    load_param_file_to_vector("../../../../../test_data/ape_params/weights.txt", weights_raw);
    load_param_file_to_vector("../../../../../test_data/ape_params/biases.txt", biases_raw);
    load_param_file_to_vector("../../../../../test_data/ape_params/scale_params.txt", sc_params_raw);
    load_param_file_to_vector("../../../../../test_data/ape_params/scale_shifts.txt", sc_shifts_raw);


    // 4. Feed Parameters into DUT Streams
    // ------------------------------------------------------------------------------
    std::cout << "[TB] Writing parameters to streams..." << std::endl;

    for (int w : weights_raw) weight_stream.write((ap_int<W_BITW>)w);
    for (int b : biases_raw) bias_stream.write((bias_T)b);

    if (sc_params_raw.size() != sc_shifts_raw.size()) {
        std::cerr << "Error: Mismatch in scale params and shifts count." << std::endl;
        return 1;
    }
    for (size_t i = 0; i < sc_params_raw.size(); ++i) {
        scale_T s;
        s.param = (ap_uint<16>)sc_params_raw[i];
        s.shift = (ap_uint<6>)sc_shifts_raw[i];
        scale_param_stream.write(s);
    }



//    dut.read_weight_from_stream(weight_stream, scale_param_stream, bias_stream);


    // 5. Load Input Data into 3D Arrays
    // ------------------------------------------------------------------------------
    std::cout << "[TB] Loading input data into 3D arrays..." << std::endl;

    // Load Y (Latent Input)
    load_file_to_3d_array<int, H, W, C_y>("../../../../../test_data/ape_input/y.txt", Y_DATA, TOTAL_Y_ELEMENTS);

    // Load Global Params (Add Stream Input)
    load_file_to_3d_array<int, H, W, C_gp>("../../../../../test_data/ape_input/global_params.txt", GP_DATA, TOTAL_GP_ELEMENTS);

    // load golden symbols
    load_file_to_3d_array<int, H, W, C_y>("../../../../../test_data/ape_golden/symbols_golden.txt", GOLDEN_SYMBOLS, TOTAL_Y_ELEMENTS);

    // load golden indices
	load_file_to_3d_array<int, H, W, C_y>("../../../../../test_data/ape_golden/scales_indices_golden.txt", GOLDEN_PDF_INDICES, TOTAL_Y_ELEMENTS);

    std::cout << "[TB] Input Dimensions: " << W << "x" << H << std::endl;

    // 6. Pack Data Streams from 3D Arrays
    // ------------------------------------------------------------------------------
    // Stream order: H -> Groups(2*SCM) -> W/2

    for (int h = 0; h < H_cosim; ++h) {
        // Iterate over Output Groups of the Entropy Layer (2*SCM)
        for (int g = 0; g < 2 * SCM; ++g) {
            // Iterate over width (2 pixels at a time)
            for (int w = 0; w < W; w += 2) {

                // --- A. Feed Global Params (Add Stream) for Layer 1 (C_gp = 2*SCM*PCN) ---
                AggData<ap_int<A_BITW>, PCN> add1, add2;
                int gp_base_idx = g * PCN;

                for (int c = 0; c < PCN; ++c) {
                    // Global Params are indexed via H[h], W[w], C[gp_base + c]
                    add1.data[c] = (ap_int<A_BITW>)GP_DATA[h][w][gp_base_idx + c];
                    add2.data[c] = (ap_int<A_BITW>)GP_DATA[h][w + 1][gp_base_idx + c];
                }
                add_stream1.write(add1);
                add_stream2.write(add2);

                // --- B. Feed Latent Y (Latent Stream) for Layer 2 (C_y = SCM*PCN) ---
                AggData<ap_int<A_BITW>, PCN> lat_pkg;
                int y_base_idx = g * (PCN / 2);

                for (int i = 0; i < PCN / 2; ++i) {
                    // Lower Half (0 to PCN/2 - 1): Pixel w
                    lat_pkg.data[i] = (ap_int<A_BITW>)Y_DATA[h][w][y_base_idx + i];

                    // Upper Half (PCN/2 to PCN - 1): Pixel w+1
                    lat_pkg.data[i + PCN / 2] = (ap_int<A_BITW>)Y_DATA[h][w + 1][y_base_idx + i];
                }
                latent_stream.write(lat_pkg);
            }
        }
    }

    // 7. Run Inference
    // ------------------------------------------------------------------------------
    std::cout << "[TB] Running Inference..." << std::endl;
    ape_top(1,H_cosim,weight_stream, scale_param_stream, bias_stream,
    		latent_stream, add_stream1, add_stream2, symbol_stream, pdf_index_stream);

    // 8. Collect and Save Outputs (Symbols and PDF Indices)
    // ------------------------------------------------------------------------------
    std::cout << "[TB] Inference done. Reordering and saving outputs..." << std::endl;

    // Stream output order: H -> Groups(2*SCM) -> W/2
    for (int h = 0; h < H_cosim; ++h) {
        for (int g = 0; g < 2 * SCM; ++g) {
            for (int w_chunk = 0; w_chunk < W / 2; ++w_chunk) {
                if (symbol_stream.empty() || pdf_index_stream.empty()) {
                    std::cerr << "Error: Unexpected empty stream at h=" << h << std::endl;
                    exit(1);
                }

                AggData<ap_int<8>, PCN> sym = symbol_stream.read();
                AggData<ap_int<8>, PCN> pdf = pdf_index_stream.read();

                int c_base = g * (PCN / 2);
                int w_even = 2 * w_chunk;
                int w_odd = 2 * w_chunk + 1;

                // Process Even Pixel (Lower Half)
                for (int i = 0; i < PCN / 2; ++i) {
                    OUTPUT_SYMBOLS[h][w_even][c_base + i] = (int)sym.data[i];
                    OUTPUT_PDF_INDICES[h][w_even][c_base + i] = (int)pdf.data[i];
                }

                // Process Odd Pixel (Upper Half)
                for (int i = 0; i < PCN / 2; ++i) {
                    OUTPUT_SYMBOLS[h][w_odd][c_base + i] = (int)sym.data[i + PCN / 2];
                    OUTPUT_PDF_INDICES[h][w_odd][c_base + i] = (int)pdf.data[i + PCN / 2];
                }
            }
        }
    }

    // 9. Write Outputs to file (Flattened HWC order)
    // ------------------------------------------------------------------------------
//    std::ofstream out_sym_file("../../../../output_symbols_hls.txt");
//    std::ofstream out_pdf_file("../../../../output_indices_hls.txt");
//
//    if (!out_sym_file.is_open() || !out_pdf_file.is_open()) {
//        std::cerr << "Error creating output files." << std::endl; return 1;
//    }
    long symbol_mismatch=0;
    long indice_mismatch=0;
    long total_elements=H_cosim*W*C_y;
    // Iterate H, W, then C (HWC layout)
    for (int h = 0; h < H_cosim; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C_y; ++c) {
//                out_sym_file << OUTPUT_SYMBOLS[h][w][c] << "\n";
//                out_pdf_file << OUTPUT_PDF_INDICES[h][w][c] << "\n";
                if(OUTPUT_SYMBOLS[h][w][c]!=GOLDEN_SYMBOLS[h][w][c])symbol_mismatch++;
                if(OUTPUT_PDF_INDICES[h][w][c]!=GOLDEN_PDF_INDICES[h][w][c])indice_mismatch++;
            }
        }
    }

//    out_sym_file.close();
//    out_pdf_file.close();

    std::cout << "[TB] Indices Mismatches:" <<indice_mismatch<<" out of "<<total_elements<<" Elements"<< std::endl;
    std::cout << "[TB] Symbol Mismatches:" <<symbol_mismatch<<" out of "<<total_elements<<" Elements"<< std::endl;
    std::cout << "[TB] Done." << std::endl;
    std::cout << "     Symbols and Indexes saved in flattened HWC format." << std::endl;

    return 0;
}
