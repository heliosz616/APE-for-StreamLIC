# Autoregressive Prediction Engine For StreamLIC

This is an HLS implementation of the Autoregressive Prediction Engine in the VCIP 2025 paper "StreamLIC: A Lightweight Learned Image Compression Model for Stream Processing On FPGA".

## File Structure

```text
.
├── APE_KERNEL/                            # HLS kernel and testbench code
│   ├── src/                               # HLS Kernel and testbench source code
│   └── ape_kernel/                        # HLS project files
└── test_data/
    ├── ape_input/                         # Input data folder for HLS testbench
    ├── ape_golden/                        # Golden reference data folder for HLS testbench
    ├── ape_params/                        # Model parameters folder for HLS testbench
    ├── params_for_quantized_inference.pt  # 12-bit quantized model parameters
    ├── perform_quantized_inference.py     # Generates TB input & golden reference
    ├── output_entropy_model_parameters.py # Extract entropy model parameters from params_for_quantized_inference.pt and rewrite as .txt for HLS testbench.
    └── test_input_image.png
```

## Getting Started

Follow these steps to generate the necessary test data and run the HLS simulation.

### Prerequisites
*   **Vitis HLS 2022.1**
*   **Python 3.x** with **PyTorch** installed

### 1. Generate Testbench Data
Before running the HLS simulation, you must generate the input parameters, feature maps, and golden reference data using the provided Python scripts.

Navigate to the `test_data` directory and run the following:

```bash
cd test_data

# Step 1: Generate model parameters for the HLS testbench
python output_entropy_model_parameters.py

# Step 2: Generate input feature maps and bit-exact golden reference
python perform_quantized_inference.py
```

### 2. Run HLS Simulation
Once the data is generated:

1.  Launch **Vitis HLS 2022.1**.
2.  Open the project located in `APE_KERNEL/ape_kernel`.
3.  Run the flow in the following order:
    *   **C Simulation** 
    *   **C Synthesis**
    *   **C/RTL Co-simulation**
## Resource Utilization

| LUT | FF | DSP | BRAM |
| :---: | :---: | :---: | :---: |
| 31,431 | 23,613 | 337 | 273 |
```
