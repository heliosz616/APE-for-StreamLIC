#ifndef _NN_HPP_
#define _NN_HPP_



#include "util.hpp"
#include "common.hpp"
#include "ape.hpp"



extern "C"{

void ape_top(
		ap_uint<16> frame_num,
		ap_uint<16> height,
		//		PARAMETERS
		hls::stream<ap_int<12>> &weight_stream,
		hls::stream<scale_T> &scale_stream,
		hls::stream<bias_T> &bias_stream,
		//		IN/OUT DATA
		hls::stream<AggData<ap_int<12>,12>> &latent_stream,
		hls::stream<AggData<ap_int<12>,12>> &add_stream1,
		hls::stream<AggData<ap_int<12>,12>> &add_stream2,
		hls::stream<AggData<ap_int<8>,12>> &symbol_stream,
		hls::stream<AggData<ap_int<8>,12>> &pdf_index_stream

		);

}


#endif


