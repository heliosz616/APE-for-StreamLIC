import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

# ==========================================
# 1. Quantized Operators
# ==========================================

ACTIVATION_BIT_WIDTH = 12
MAX_VALUE=(2**(ACTIVATION_BIT_WIDTH - 1)-1)
MIN_VALUE=-(2**(ACTIVATION_BIT_WIDTH - 1)-1)


def fgp_conv(q_in, params, stride, padding, groups=1):
    """
    Performs a quantized 2D convolution with Sparsity Mask.
    Operation: Output = ((Input * (Weight * Mask) + Bias) * Scale) >> Shift
    """

    q_in = q_in.to(torch.int64)

    # 1. Apply Mask to Weights (Element-wise)
    q_weight = params['weight'] * params['mask']
    q_bias = params.get('bias')
    scale_param = params['scale_param'].view(1, -1, 1, 1)
    scale_shift = params['scale_shift'].view(1, -1, 1, 1)

    # 2. Integer Convolution (Accumulation)
    # Result is a large integer (approx 32-bit or more)
    accumulator = F.conv2d(q_in.double(), q_weight.double(), stride=stride, padding=padding, groups=groups).round().to(
        torch.int64)

    # 3. Apply Per-Channel Scaling
    # Multiplies accumulator by scale_param, then right shifts.
    # This emulates the floating point scaling: acc * (S_in * S_w / S_out)
    scaled_accumulator = ((accumulator * scale_param).to(torch.int64)) >> scale_shift

    # 4. Add Quantized Bias
    # The bias provided in params is already scaled to match the output domain.
    if q_bias is not None:
        scaled_accumulator += q_bias.view(1, -1, 1, 1)

    # 5. Final Down-shift to Activation Bit Width
    # Fits the result into the target quantization range (e.g., 12-bit)
    res_bits = 16 - ACTIVATION_BIT_WIDTH
    HALF_VALUE=1<<(res_bits-1)
    q_out = (scaled_accumulator+HALF_VALUE) >> res_bits

    return torch.clamp(q_out.to(torch.int32), MIN_VALUE, MAX_VALUE)


def quantized_conv2d(q_in, params, stride, padding, groups=1):
    """Standard Quantized Conv2d without Mask."""
    q_in = q_in.to(torch.int64)

    q_weight = params['weight']
    q_bias = params.get('bias')
    scale_param = params['scale_param'].view(1, -1, 1, 1)
    scale_shift = params['scale_shift'].view(1, -1, 1, 1)

    accumulator = F.conv2d(q_in.double(), q_weight.double(), stride=stride, padding=padding, groups=groups).round().to(
        torch.int64)

    scaled_accumulator = ((accumulator * scale_param).to(torch.int64)) >> scale_shift

    if q_bias is not None:
        scaled_accumulator += q_bias.view(1, -1, 1, 1)

    res_bits = 16 - ACTIVATION_BIT_WIDTH
    HALF_VALUE = 1 << (res_bits - 1)
    q_out = (scaled_accumulator + HALF_VALUE) >> res_bits

    return torch.clamp(q_out.to(torch.int32), MIN_VALUE, MAX_VALUE)


def quantized_lagc(q_in, params):
    """
    Quantized LAGC (Local Adaptive Gain Control).
    Path: Conv1x1 -> Tanh(LUT) -> Gain -> Output Scale
    """
    q_in = q_in.to(torch.int64)

    # 1. Affine Convolution
    affine_params = {
        'weight': params['affine_weight'],
        'bias': params['affine_bias'],
        'scale_param': params['affine_scale_param'],
        'scale_shift': params['affine_scale_shift']
    }
    tanh_in_q = quantized_conv2d(q_in, affine_params, stride=1, padding=0)
    tanh_in_q.clamp_(-2047, 2047)

    # 2. Tanh Lookup Table (LUT)
    # Training Scale: Input 1/256 -> Tanh -> Output 1/1024
    # We construct a LUT mapping integer inputs [-2047, 2047] to quantized tanh outputs.
    # range logic: -2047/256 = -7.99 (tanh is -1), 2047/256 = 7.99 (tanh is 1).
    tanh_lut = torch.round(torch.tanh(torch.arange(-2047, 2048) / 256.0) * 1024.0).to(torch.int32).to(q_in.device)

    # Offset input by 2047 to use as array index (0 to 4094)
    tanh_out_q = tanh_lut[tanh_in_q + 2047].to(torch.int64)

    # 3. Gain Calculation
    # gain = (tanh_out * score_scale) + score_bias
    score_scale_param = params['score_scale_param'].view(1, -1, 1, 1)
    score_scale_shift = params['score_scale_shift'].view(1, -1, 1, 1)
    score_bias = params['score_bias'].view(1, -1, 1, 1)


    gain_q_intermediate = (tanh_out_q * score_scale_param) >> score_scale_shift
    gain_q_biased = gain_q_intermediate + score_bias

    res_bits = 16 - ACTIVATION_BIT_WIDTH
    HALF_VALUE = 1 << (res_bits - 1)
    gain_q = (gain_q_biased+HALF_VALUE) >> res_bits
    gain_q=torch.clamp(gain_q, MIN_VALUE, MAX_VALUE)

    # 4. Final Output Scaling (Element-wise Multiplication)
    output_scale_param = params['output_scale_param'].view(1, -1, 1, 1)
    output_scale_shift = params['output_scale_shift'].view(1, -1, 1, 1)

    # We shift the input left by 16 to maintain precision during multiplication
    intermediate_precision_shift = 16
    q_out_unscaled = (q_in << intermediate_precision_shift) * gain_q

    final_shift = output_scale_shift + intermediate_precision_shift
    q_out = (q_out_unscaled * output_scale_param) >> final_shift

    return torch.clamp(q_out.to(torch.int32), MIN_VALUE, MAX_VALUE)


def quantized_leaky_relu(q_in):
    """LeakyReLU with negative_slope=1/64, returning Integers."""
    # Simulation: Convert to float, apply exact PyTorch logic, round back to Int
    # float_in = q_in.to(torch.float32)
    # out = torch.round(F.leaky_relu(float_in, negative_slope=1.0 / 64.0))
    # return out.to(torch.int32)
    out = q_in.clone()

    # 2. Apply shift logic for negative numbers
    # Note: In Python, >> on negative numbers is arithmetic shift (like C++),
    # so -64 >> 6 = -1. This matches HLS.
    mask = q_in < 0
    out[mask] = (q_in[mask] + 32) >> 6

    return out


def quantized_upsample(q_in, params):
    """Conv + PixelShuffle + Crop."""
    q_in_padded = F.pad(q_in, (1, 1, 1, 1))
    q_conv_out = quantized_conv2d(q_in_padded, params, stride=1, padding=0)
    q_shuffled = F.pixel_shuffle(q_conv_out, 2)
    q_out = q_shuffled[:, :, 1:-1, 1:-1]
    return q_out


def quantized_channel_addition(x):
    """Split, Add, Clamp."""
    x1, x2 = x.chunk(2, dim=1)
    # Integer clamp limit for 12-bit signed: +/- 2047
    return torch.clamp(x1 + x2, min=-2047, max=2047)


def save_integer_txt(tensor, filename):
    """
    Reshapes (B, C, H, W) -> (H, W, C) -> Flattened (H*W, C)
    Saves as INT (no decimals).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove Batch

    # Permute to (H, W, C)
    # This ensures that when we flatten, we iterate Width, then Height (Row-Major)
    tensor_hwc = tensor.permute(1, 2, 0)

    # Flatten to (H*W, C)
    flat_data = tensor_hwc.flatten().cpu().numpy()

    # Save as integers
    np.savetxt(filename, flat_data, fmt='%d')
    print(f"Saved {filename} (Shape: {tensor_hwc.shape}, Type: Integer)")


# ==========================================
# 2. Main Inference Logic
# ==========================================

def run_inference(image_path, params_path):
    device = torch.device("cpu")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"{params_path} not found.")

    all_params = torch.load(params_path, map_location=device)
    print("Parameters loaded.")

    # Load Image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((768,1280))])
    # Input is 0-255 Integer
    x = (transform(img) * 510.0).round().unsqueeze(0).to(torch.int32).to(device)-255
    print(f"Image shape: {x.shape}")

    # -----------------------------------------------------------
    # Analysis Transform (g_a)
    # -----------------------------------------------------------
    x = quantized_conv2d(x, all_params['g_a.0'], stride=2, padding=1)
    x = quantized_lagc(x, all_params['g_a.1'])
    x = quantized_conv2d(x, all_params['g_a.2'], stride=2, padding=1)
    x = quantized_lagc(x, all_params['g_a.3'])
    x = quantized_conv2d(x, all_params['g_a.4'], stride=2, padding=1)
    x = quantized_lagc(x, all_params['g_a.5'])
    x = quantized_conv2d(x, all_params['g_a.6'], stride=2, padding=1)

    # Latent y (Fixed Point, Scale 1/16)
    y = torch.clamp(x, min=-127 * 16, max=127 * 16)

    # SAVE Y (Integer)
    save_integer_txt(y, "./ape_input/y.txt")

    # -----------------------------------------------------------
    # Hyper Analysis (h_a)
    # -----------------------------------------------------------
    z_in = F.pad(y, (1, 1, 2, 0))
    z_feat = quantized_conv2d(z_in, all_params['h_a.1'], stride=2, padding=0)
    z_feat = quantized_lagc(z_feat, all_params['h_a.2'])
    z = quantized_conv2d(z_feat, all_params['h_a.3'], stride=2, padding=1)

    # -----------------------------------------------------------
    # Hyper Latent Quantization
    # -----------------------------------------------------------
    # z is Fixed Point (Scale 1/16).
    # z_hat = Round(z / 16)
    z_hat = (z + 8) >> 4
    z_hat = torch.clamp(z_hat, -127, 127)

    # SAVE Z_HAT (Integer)
    save_integer_txt(z_hat, "z_hat.txt")

    # -----------------------------------------------------------
    # Hyper Synthesis (h_s) -> Global Params
    # -----------------------------------------------------------
    gs_out = quantized_upsample(z_hat, all_params['h_s.0'])
    gs_out = quantized_leaky_relu(gs_out)
    global_params = quantized_upsample(gs_out, all_params['h_s.2'])

    # SAVE GLOBAL PARAMS (Integer)
    save_integer_txt(global_params, "./ape_input/global_params.txt")
    print(global_params.shape)

    # -----------------------------------------------------------
    # Autoregressive Reconstruction Loop
    # -----------------------------------------------------------
    B, C, H, W = y.shape

    # Context Buffer: Padded by (1,1,3,0)
    y_hat_pad = F.pad(torch.zeros_like(y), (1, 1, 3, 0))
    y_hat_final = torch.zeros_like(y)
    symbol_storage=torch.zeros_like(y, dtype=torch.int32)

    # Buffers for saving Integer Stats
    means_storage = torch.zeros_like(y, dtype=torch.int32)
    scales_storage = torch.zeros_like(y, dtype=torch.int32)

    print("Starting autoregressive loop...")
    for h in range(H):
        # 1. Extract Context (3 rows above current)
        ctx_input = y_hat_pad[:, :, h:h + 3, :].clone()

        # 2. Context Prediction Network
        ctx_feat = fgp_conv(ctx_input, all_params['context_prediction.1'], stride=1, padding=0)
        ctx_feat = quantized_leaky_relu(ctx_feat)
        ctx_p = fgp_conv(ctx_feat, all_params['context_prediction.3'], stride=1, padding=(0, 1))

        # 3. Concatenate with Global Params
        p = global_params[:, :, h:h + 1, :]
        combined = torch.cat((p, ctx_p), dim=1)

        # 4. Entropy Parameters Network
        ep_feat = quantized_channel_addition(combined)
        ep_feat = quantized_leaky_relu(ep_feat)
        gaussian_params = fgp_conv(ep_feat, all_params['entropy_parameters.2'], stride=1, padding=(0, 1))

        # 5. Extract Means and Scales (Integers)
        # These are fixed-point values.
        # means_hat scale: 1/16
        # scales_hat scale: Raw values (used as exponents)
        means_hat = gaussian_params[:, 0::2, :, :]
        scales_hat = torch.clamp(gaussian_params[:, 1::2, :, :],min=0,max=127)

        # 6. Reconstruct Slice (De-quantization / Inverse Prediction)
        # Algorithm:
        # Residual = Y - Mean
        # Index = Round(Residual / Step)  (Step is 1.0, but in fixed-point x16 it is 16)
        # Y_hat = (Index * Step) + Mean

        curr_y = y[:, :, h:h + 1, :]

        # Calculate residual in fixed-point
        residue_fixed = curr_y - means_hat

        # Convert fixed-point residual to integer index (divide by 16 with rounding)
        # (x + 8) >> 4 performs rounding for positive numbers.
        # For signed numbers, this is a valid floor((x+8)/16) approx for rounding.
        residue_index = (residue_fixed + 8) >> 4
        residue_index = torch.clamp(residue_index, -127, 127)

        # Reconstruct back to fixed-point
        reconstructed_slice = (residue_index << 4) + means_hat

        # Update buffers
        y_hat_pad[:, :, h + 3:h + 4, 1:-1] = reconstructed_slice
        y_hat_final[:, :, h:h + 1, :] = reconstructed_slice
        symbol_storage[:, :, h:h + 1, :] = residue_index

        # Store stats
        means_storage[:, :, h:h + 1, :] = means_hat
        scales_storage[:, :, h:h + 1, :] = scales_hat

    # SAVE AUTOREGRESSIVE STATS (Integers)
    save_integer_txt(symbol_storage, "./ape_golden/symbols_golden.txt")
    save_integer_txt(scales_storage, "./ape_golden/scales_indices_golden.txt")

    # -----------------------------------------------------------
    # Synthesis Transform (g_s)
    # -----------------------------------------------------------
    # Input: y_hat (Fixed Point, Scale 1/16)
    x_rec = quantized_upsample(y_hat_final, all_params['g_s.0'])
    x_rec = quantized_lagc(x_rec, all_params['g_s.1'])
    x_rec = quantized_upsample(x_rec, all_params['g_s.2'])
    x_rec = quantized_lagc(x_rec, all_params['g_s.3'])
    x_rec = quantized_upsample(x_rec, all_params['g_s.4'])
    x_rec = quantized_lagc(x_rec, all_params['g_s.5'])
    x_rec = quantized_upsample(x_rec, all_params['g_s.6'])

    # -----------------------------------------------------------
    # Output Reconstruction (Final Float Convert)
    # -----------------------------------------------------------
    # We need the float output scale of the last layer to map the
    # integer output x_rec back to the [0, 255] domain.
    final_scale = all_params['g_s.6']['float_output_scale'].item()

    # Formula: FloatVal = (IntVal * Scale) + Bias(0.5)
    x_float = (x_rec.float() * final_scale) + 0.5

    x_clamped = torch.clamp(x_float * 255.0, 0, 255).to(torch.uint8)

    img_out = transforms.ToPILImage()(x_clamped.squeeze(0).cpu())
    img_out.save("output_recon.png")
    print("Reconstructed image saved to output_recon.png")


if __name__ == "__main__":


    run_inference("./test_input_image.png", "./params_for_quantized_inference.pt")
