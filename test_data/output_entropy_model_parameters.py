import torch
import numpy as np
import os
import sys

# ==========================================
# Configuration matching HLS Template
# ==========================================
PCN = 12
M = 192  # Base channel count
SCM = M // PCN  # 16
BN_CONFIG = {
    'context_prediction.1': 27,  # BN1
    'context_prediction.3': 9,  # BN2
    'entropy_parameters.2': 12  # BN3
}


def get_hls_params(layer_name, all_params):
    """
    Extracts raw tensors for a specific layer from the nested dictionary.
    Structure: all_params[layer_name][param_key]
    Returns: masked_weight, bias, scale_param, scale_shift, mask
    """

    # 1. Access the specific layer's parameter dictionary
    if layer_name not in all_params:
        print(f"Error: Layer '{layer_name}' not found in integer_params.pt")
        print("Available keys:", list(all_params.keys()))
        sys.exit(1)

    layer_params = all_params[layer_name]

    # 2. Extract Weights
    # Ensure float for calculation, though they are stored as int32/float in PT
    weight = layer_params['weight'].float()

    # 3. Extract Mask
    # Mask is stored in the params dict for FGPConv layers
    if 'mask' not in layer_params:
        print(f"Error: 'mask' not found for layer {layer_name}. Is it an FGPConv layer?")
        sys.exit(1)

    mask = layer_params['mask'].float()
    # Ensure binary mask (0 or 1)
    mask = (mask > 0.5).float()

    # 4. Extract Bias
    if 'bias' in layer_params and layer_params['bias'] is not None:
        bias = layer_params['bias'].int()
    else:
        # Create zero bias if missing
        bias = torch.zeros(weight.shape[0], dtype=torch.int32)

    # 5. Extract Scales
    s_param = layer_params['scale_param'].int()
    s_shift = layer_params['scale_shift'].int()

    return weight * mask, bias, s_param, s_shift, mask


def export_to_hls_files():
    params_path = "./params_for_quantized_inference.pt"
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found.")
        return

    # Load the nested dictionary
    all_params = torch.load(params_path, map_location='cpu')

    # Storage for HLS streams
    stream_weights = []
    stream_ch_ptr = []
    stream_offsets = []
    stream_offsets=[]
    stream_biases = []
    stream_scale_params = []
    stream_scale_shifts = []

    # Layer Definitions: (Name, Kernel Height, Kernel Width)
    layers = [
        ('context_prediction.1', 3, 3),
        ('context_prediction.3', 1, 3),
        ('entropy_parameters.2', 1, 3)
    ]

    print("----------------------------------------------------------------")
    print("Exporting HLS parameters with Mask Verification")
    print("----------------------------------------------------------------")




    # ---------------------------------------------------------
    # 2. Process Each Layer
    # ---------------------------------------------------------
    for layer_name, Kh, Kw in layers:
        target_BN = BN_CONFIG[layer_name]
        print(f"\nProcessing Layer: {layer_name}")

        # Extract Tensors using the corrected access pattern
        w, b, sp, ss, mask = get_hls_params(layer_name, all_params)

        C_out, C_in, _, _ = w.shape
        SCM_out = C_out // PCN
        SCM_in = C_in // PCN

        # --- A. Export Scales & Biases ---
        # Ordering: Group 0..2*SCM -> Channel 0..PCN
        sp_g = sp.view(SCM_out, PCN)
        ss_g = ss.view(SCM_out, PCN)
        b_g = b.view(SCM_out, PCN)

        for g in range(SCM_out):
            for c in range(PCN):
                stream_scale_params.append(sp_g[g, c].item())
                stream_scale_shifts.append(ss_g[g, c].item())
                stream_biases.append(b_g[g, c].item())

        # HLS expects 128 bias entries.
        # Format: High Byte = 127, Low Byte = -127


        # --- B. Export Weights using Mask ---

        # 1. Reshape Mask to Block View
        # Original: (C_out, C_in, Kh, Kw) -> (SCM_out, PCN, SCM_in, PCN, Kh, Kw)
        mask_groups = mask.view(SCM_out, PCN, SCM_in, PCN, Kh, Kw)
        w_groups = w.view(SCM_out, PCN, SCM_in, PCN, Kh, Kw)
        print(torch.max(w))

        # 2. Permute to Traversal Order: (SCM_out, Kh, Kw, SCM_in, PCN, PCN)
        # HLS Order: For each Output Group, traverse Kh -> Kw -> Input Group
        mask_blocks = mask_groups.permute(0, 4, 5, 2, 1, 3)
        w_blocks = w_groups.permute(0, 4, 5, 2, 1, 3)

        total_active_blocks = 0

        # Iterate over Output Channel Groups
        for og in range(SCM_out):
            group_active_count = 0

            # Traversal Order: Kernel Height -> Kernel Width -> Input Channel Group
            for kh in range(Kh):
                for kw in range(Kw):
                    for ig in range(SCM_in):

                        # Extract the mask for this block (PCN x PCN)
                        blk_mask = mask_blocks[og, kh, kw, ig, :, :]

                        # Check if block is active (sum of mask > 0)
                        if torch.sum(blk_mask) > 0:
                            if torch.sum(blk_mask)!=PCN*PCN:
                                print("Sparse block mismatch")
                                sys.exit(1)

                            group_active_count += 1
                            total_active_blocks += 1

                            # --- Export Metadata ---

                            # Channel Pointer Calculation
                            # Layer 1 (3x3): Flattened index implies y-offset + channel-offset
                            if "context_prediction.1" in layer_name:
                                ch_ptr = kh * SCM + ig
                            else:  # 1x3 Conv
                                ch_ptr = ig

                            horiz_offset = kw

                            stream_offsets.append(ch_ptr)
                            stream_offsets.append(horiz_offset)
                            print(layer_name,ch_ptr,horiz_offset)


                            # --- Export Weight Data ---
                            blk_weight = w_blocks[og, kh, kw, ig, :, :]
                            blk_data_np = blk_weight.int().numpy()

                            # Flatten row-major
                            for r in range(PCN):
                                for c in range(PCN):
                                    stream_weights.append(blk_data_np[r, c])

            # --- Verification ---
            if group_active_count != target_BN:
                print(f"  [Error] Output Group {og}: Found {group_active_count} active blocks, expected {target_BN}.")
                sys.exit(1)

        print(f"  Verified: {SCM_out} groups each have {target_BN} active blocks.")

    # ---------------------------------------------------------
    # 3. Write Output Files
    # ---------------------------------------------------------
    def write_txt(filename, data_list):
        with open(filename, 'w') as f:
            for val in data_list:
                f.write(f"{int(val)}\n")
        print(f"Saved {filename}: {len(data_list)} lines")

    print("Storing offsets into bias stream")
    stream_biases.extend(stream_offsets)

    print("Generating PDF bounds header for bias stream...")
    for i in range(128):
        stream_biases.append(127)
        stream_biases.append(-127)

    print("\nWriting HLS files...")
    write_txt('./ape_params/weights.txt', stream_weights)
    write_txt('./ape_params/biases.txt', stream_biases)
    write_txt('./ape_params/scale_params.txt', stream_scale_params)
    write_txt('./ape_params/scale_shifts.txt', stream_scale_shifts)
    print("Export Complete.")


if __name__ == "__main__":
    export_to_hls_files()