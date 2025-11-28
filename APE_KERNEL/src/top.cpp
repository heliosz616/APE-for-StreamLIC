
#include "top.hpp"






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

		){

	static APEngine<12,12,40,12,16,27,9,12,80> ep;
	ep.init();
	ep.read_weight_from_stream(weight_stream,scale_stream,bias_stream);

	ape_frame_loop:
	for(int i=0;i<frame_num;i++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=40
		ep.run(height,
		latent_stream,
		add_stream1,add_stream2,
		symbol_stream,pdf_index_stream);
	}

}









