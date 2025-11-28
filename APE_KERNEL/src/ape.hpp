#ifndef _CTXT_HPP_
#define _CTXT_HPP_

#include "common.hpp"
#include <iostream>


#define w_T ap_int<w_bitw>
#define w_wg31 ap_int<w_bitw+2>
#define a_T ap_int<a_bitw>
#define out_T ap_int<o_bitw>
#define buf_T ap_int<bf_bitw>


/*
	The following model performs 3 layers of autoregressive sparse convolution inference
	The input channel number of layer 1 is SCM*PCN
	The out_channel numbers of every layer are 2*SCM*PCN
	PCN is the number of channels processed in parallelim.
	Parallelism applied on 3 dimensions:input channel(PCN), output channel(PCN),output width(2)
	Total parallelism is 2*PCN*PCN
	Grouping PCN input channel and PCN output channel into a block,each time a block of weights is multiply accumulated with the input.
	Such grouping allows a block sparse inference, When using the block weights, and its corresponding channel pointer and spatial offset

	First layer convolution kernel size 3x3, its input stored in activation_buffer[2].
	The second dimension of activation buffer corresponding to channel dimension, or flattened height and channel dimension

	Second layer convolution kernel size (1x3),
	The result of the second layer is added to add_stream, then applied relu activation.

	The third layer reads the output of the second layer, kernel size(1x3)
	It outputs mean and distrubution scale. It also read latent stream of y, quantize it, and store dequantized y_hat=round(y-mean)+mean into activation_buffer[2], which is the context input of the next iteration.
	It also outputs symbol s=round(y-means) into sybol_stream together with scale parameter into pdf_index_stream
*/

template< unsigned short w_bitw, int a_bitw, int bf_bitw,
int PCN,int SCM,int BN1,int BN2,int BN3,int MAX_W>
class APEngine {
private:
//  PCN=k*PCN, k>0
	//M =PCN*SCM
	//input channel number: M
	//output channel number of layer1 2M
	//output channel number of layer2 2M
	//output channel number of layer3 2M:M channels for means, M channels for scales, means is used for autoregressive quantization:s=round(y-means), y_hat=round(y-means)+means
	static constexpr unsigned short total_layer_num=3;
	static constexpr unsigned short WGT_CATO=4;//concated parameter to accelerate weight block loading process

	//Every layer has 2*SCM output channel groups, each output channel group has BN1,BN2,BN3 weight blocks in layer0,layer 1,and layer 2
	static constexpr unsigned int WB_NUM=(2*SCM)*BN3+(2*SCM)*BN2+2*SCM*BN1;
	const ap_uint<8> block_num[3]={BN1,BN2,BN3};//default 27,9,12


//	A 3-bank buffer for inter layer activation pingpong access.
//	activation_buffer[0] and activation_buffer[1] are conventional activation buffer
//	activation_buffer[2] is used to store 3 rows and SCM*PCN channels of historic input data for autoregressive prediction
//	After each iteration activation_buffer[2] update the oldest row with new data
	AggData<a_T,PCN> activation_buffer[3][3*SCM][MAX_W+2];


	//channel_ptr point to the corresponding input channel for a specific weight block
	//The channel_ptr of the 2nd and 3rd layer points out  channel dimension c, ptr=c
	//The channel_ptr of the first layer also points height dimension y, ptr=y*SCM+c. In activation_buffer[2], channel and height dimension are flattened.
	ap_uint<12> channel_ptr[WB_NUM];

	// point to horizontal offset corresponding to a weight block,<3
	ap_uint<4> horizontal_offset[WB_NUM];

	ap_uint<16> block_ptr_idx;//point to the horizontal_offset and channel_ptr for the current weight block for input data accessing
	ap_uint<16> wb_index;//point to the current weight block


	AggData<w_T, PCN*WGT_CATO> weight_buffer[WB_NUM*PCN/WGT_CATO];


// bias and quantization parameters
	static constexpr unsigned int TOTAL_SOCN_NUM=2*SCM+2*SCM+2*SCM;
	AggData<scale_T,PCN> scales[TOTAL_SOCN_NUM];
	AggData<bias_T,PCN> bias[TOTAL_SOCN_NUM];
	ap_uint<16> scale_bias_index;



//	read and write pointers. Read pointer is accessed in layer 0, altered in layer1, write pointer is accessed and altered in layer2
	ap_uint<12> first_layer_start_channel_index=0;// A read pointer, point to the top line (oldest) in the rotation buffer activation_buffer[2], which is the channel pointer offset for block sparse accumulation.
	ap_uint<4> input_line_ptr;// A write pointer, point to the oldest line in the rotation buffer activation_buffer[2] to be updated by new data


// The output symbols need to be clamped using the following parameters
	ap_int<8> symbol_max_value[128];
	ap_int<8> symbol_min_value[128];

// A pingpong weight register pile
	w_T weight_reg[2][PCN][PCN];
	bool weight_pingpong;

//	A pingpong psum buffer
	AggData<buf_T, PCN> mac_buffer[2][MAX_W];


public:
	APEngine(){
		#pragma HLS AGGREGATE compact=auto variable=weight_buffer
		#pragma HLS AGGREGATE compact=auto variable=activation_buffer
		#pragma HLS AGGREGATE compact=auto variable=mac_buffer

		#pragma HLS AGGREGATE compact=auto variable=scales
		#pragma HLS AGGREGATE compact=auto variable=bias

		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=block_num

		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=activation_buffer
		#pragma HLS ARRAY_PARTITION dim=3 factor=2 type=cyclic variable=activation_buffer

		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=symbol_max_value
		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=symbol_min_value


		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight_reg
		#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight_reg
		#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=weight_reg


		#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=mac_buffer
		#pragma HLS ARRAY_PARTITION dim=2 factor=2 type=cyclic variable=mac_buffer
    }

    void init(){
    	AggData<a_T,PCN>act_zero_reg;
		AggInit<a_T,PCN>(act_zero_reg);
    	for(int i=0;i<3;i++){
			#pragma HLS PIPELINE off
    		for(int j=0;j<3*SCM;j++){
				#pragma HLS PIPELINE off
    			for(int k=0;k<MAX_W+2;k++){
					#pragma HLS PIPELINE off
    				activation_buffer[i][j][k]=act_zero_reg;
    			}
    		}
    	}
    }



    void read_weight_from_stream(hls::stream<w_T> &weight_stream,
    		hls::stream<scale_T> &scale_param_stream,
			hls::stream<bias_T> &bias_stream){
    	weight_buffer1_fill_loop:
    	for(int i=0;i<(WB_NUM*PCN/WGT_CATO);++i){
			#pragma HLS PIPELINE off
    		for(int j=0;j<WGT_CATO;++j){
				#pragma HLS PIPELINE off
    			for(int k=0;k<PCN;k++){
					#pragma HLS PIPELINE off
    				weight_buffer[i].data[j*PCN+k]=weight_stream.read();
    			}
    		}
    	}

    	scale_bias_fill_loop:
    	for(int i=0;i<TOTAL_SOCN_NUM;++i){
			#pragma HLS PIPELINE off
    		for(int j=0;j<PCN;++j){
				scales[i].data[j]=scale_param_stream.read();
				bias[i].data[j]=bias_stream.read();
    		}
		}

    	group_index_fill_loop:
		for(int i=0;i<WB_NUM;++i){
			#pragma HLS PIPELINE off
			channel_ptr[i]=(ap_uint<12>)(bias_stream.read());
			horizontal_offset[i]=(ap_uint<4>)(bias_stream.read());
		}

    	scale_max_read_loop:
		for(int i=0;i<128;++i){
			#pragma HLS PIPELINE off
			symbol_max_value[i] = (ap_int<8>) (bias_stream.read());
			symbol_min_value[i] = (ap_int<8>) (bias_stream.read());
		}

    }

    void load_weight_reg(w_T weight_read_reg[PCN][PCN]){
    	ap_uint<16> offset=wb_index*(PCN/WGT_CATO);
		weight_read_loop:
		for(int i=0;i<PCN/WGT_CATO;i++){
			#pragma HLS PIPELINE II=1
			AggData<w_T,PCN*WGT_CATO> read_bundle;
			#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=read_bundle.data
			read_bundle=weight_buffer[offset+i];
			for(int k=0;k<WGT_CATO;k++){
				#pragma HLS UNROLL
				for(int j=0;j<PCN;j++){
					#pragma HLS UNROLL
					weight_read_reg[i*WGT_CATO+k][j]=read_bundle.data[k*PCN+j];
				}
			}
		}
		if(wb_index==WB_NUM-1)wb_index=0;
		else wb_index++;
    }

    void multiply_accumulator(
    			w_T weight_mac_reg[PCN][PCN],
    			AggData<buf_T, PCN> mac_in[MAX_W],
				AggData<buf_T, PCN> mac_out[MAX_W],
				bool init_flag,
				bool output_flag,
        		hls::stream<AggData<a_T,PCN>> &input_stream1,
        		hls::stream<AggData<a_T,PCN>> &input_stream2,
        		hls::stream<AggData<buf_T,PCN>> &mac_out_stream1,
    			hls::stream<AggData<buf_T,PCN>> &mac_out_stream2){
			#pragma HLS INTERFACE mode=bram port=mac_in
			#pragma HLS INTERFACE mode=bram port=mac_out
    		mac_one_line_loop:
			bool store_reg_pingpong=0;
			for(int k=0;k<MAX_W/2;k++){
				#pragma HLS PIPELINE II=1
				AggData<a_T,PCN> input_reg1;
				AggData<a_T,PCN> input_reg2;
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=input_reg1.data
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=input_reg2.data

				AggData<buf_T,PCN> output_reg1;
				AggData<buf_T,PCN> output_reg2;
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=output_reg1.data
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=output_reg2.data


				input_reg1=input_stream1.read();
				input_reg2=input_stream2.read();

				if(init_flag){
					AggInit<buf_T,PCN>(output_reg1);
					AggInit<buf_T,PCN>(output_reg2);
				}
				else{
					output_reg1=mac_in[k*2];
					output_reg2=mac_in[k*2+1];
				}


				mac_outer_loop:
				for(int i=0;i<PCN;i++){
					#pragma HLS UNROLL
					mac_inner_loop:
					for(int j=0;j<PCN;j++){
						#pragma HLS UNROLL
						output_reg1.data[i]+=(buf_T)input_reg1.data[j]*weight_mac_reg[i][j];
						output_reg2.data[i]+=(buf_T)input_reg2.data[j]*weight_mac_reg[i][j];
					}
				}

				if(output_flag){
					mac_out_stream1.write(output_reg1);
					mac_out_stream2.write(output_reg2);

				}
				else{
					mac_out[k*2]=output_reg1;
					mac_out[k*2+1]=output_reg2;

				}
			}


    }

    void run_one_block(
				w_T weight_mac_reg[PCN][PCN],
				w_T weight_read_reg[PCN][PCN],
				AggData<buf_T, PCN> mac_in[MAX_W],
				AggData<buf_T, PCN> mac_out[MAX_W],
				bool init_flag,
				bool output_flag,
        		hls::stream<AggData<a_T,PCN>> &input_stream1,
        		hls::stream<AggData<a_T,PCN>> &input_stream2,
        		hls::stream<AggData<buf_T,PCN>> &mac_out_stream1,
    			hls::stream<AggData<buf_T,PCN>> &mac_out_stream2){
		#pragma HLS INTERFACE mode=bram port=mac_in
		#pragma HLS INTERFACE mode=bram port=mac_out
		#pragma HLS DATAFLOW

    	multiply_accumulator(
						weight_mac_reg,mac_in,mac_out,
						init_flag,output_flag,input_stream1,input_stream2,
						mac_out_stream1,mac_out_stream2);
		load_weight_reg(weight_read_reg);

    }

    void run_mac(
    		ap_uint<4> layer_num,
    		hls::stream<AggData<a_T,PCN>> &input_stream1,
    		hls::stream<AggData<a_T,PCN>> &input_stream2,

    		hls::stream<AggData<buf_T,PCN>> &mac_out_stream1,
			hls::stream<AggData<buf_T,PCN>> &mac_out_stream2){
    	bool mac_inport=0;
    	mac_outer_loop:
    	for(int odepth=0;odepth<2*SCM;odepth++){
    		mac_one_block_loop:
    		for(int idepth=0;idepth<block_num[layer_num];idepth++){
				#pragma HLS LOOP_TRIPCOUNT min=9 max=27 avg=16
//				#pragma HLS LOOP_TRIPCOUNT min=112 max=112 avg=112

    			bool init_flag=(idepth==0);
    			bool output_flag=(idepth==block_num[layer_num]-1);
    			run_one_block(	weight_reg[weight_pingpong],weight_reg[!weight_pingpong],
    							mac_buffer[mac_inport],	mac_buffer[!mac_inport],
    							init_flag,output_flag,
								input_stream1,input_stream2,
    			        		mac_out_stream1,mac_out_stream2);
				weight_pingpong=!weight_pingpong;
				mac_inport=!mac_inport;
    		}
    	}
    }

    void agg_bias_scale_quant(AggData<ap_int<a_bitw>,PCN> &out_reg,
    		AggData<ap_int<bf_bitw>,PCN> mac_data, AggData<scale_T,PCN> scale_reg, AggData<bias_T,PCN> bias_reg){
    	#pragma HLS INLINE
    	const ap_int<bf_bitw> MAX_A_VALUE=(1<<(a_bitw-1))-1;
    	const ap_int<16> HALF_VALUE=1<<(15-a_bitw);
    	for(int i=0;i<PCN;i++){
    		#pragma HLS UNROLL
    		ap_int<bf_bitw> scaled= (ap_int<bf_bitw>)(((ap_int<bf_bitw+16>)mac_data.data[i]*scale_reg.data[i].param)>>scale_reg.data[i].shift);
    		scaled = (scaled+bias_reg.data[i]+HALF_VALUE)>>(16-a_bitw);
    		if(scaled > MAX_A_VALUE) out_reg.data[i] = MAX_A_VALUE;
    		else if(scaled < -MAX_A_VALUE) out_reg.data[i] =-MAX_A_VALUE;
    		else out_reg.data[i]=scaled;
    	}
    }


    void QuantLatent(
    		AggData<ap_int<8>,2*(PCN/2)> &symbol,
    		AggData<ap_int<8>,2*(PCN/2)> &scale_output,
    		AggData<ap_int<a_bitw>,PCN> &dequant_reg,
			AggData<ap_int<a_bitw>,2*(PCN/2)> input,
			AggData<ap_int<a_bitw>,PCN> param_reg1,AggData<ap_int<a_bitw>,PCN> param_reg2)
    /*Use param_reg1 and param_reg 2 to quantize input into symbol, and dequantize into dequant_reg
     *
     * */
    {
    	#pragma HLS INLINE
    	const ap_int<bf_bitw> MAX_A_VALUE=(1<<(a_bitw-1))-1;
    	for(int i=0;i<PCN/2;i++)
    	{
    		#pragma HLS UNROLL
//    		symbol,input shape are 2x(PCN/2), 2 in horizontal dimension, and PCN/2 channel dimension
//    		First PCN/2 elements are quantized with param_reg1, the rest with param_reg2
    		ap_int<a_bitw> half_value=1<<(a_bitw-9);

//			get symbol_1, scale_1,latent_1
//    		means from even channels, scales from odd channels
    		ap_uint<8> scale_index1;
    		if(param_reg1.data[i*2+1]>127)scale_index1=127;
    		else if(param_reg1.data[i*2+1]<0)scale_index1=0;
    		else scale_index1=(ap_uint<8>)param_reg1.data[i*2+1];
    		ap_int<a_bitw+2> symbol1=((ap_int<a_bitw+2>)input.data[i]-param_reg1.data[i*2]+half_value)>>(a_bitw-8);
    		ap_int<a_bitw> max_value1=symbol_max_value[scale_index1];
    		ap_int<a_bitw> min_value1=symbol_min_value[scale_index1];


			ap_int<a_bitw> symbol1_clamped;
			ap_int<a_bitw+2> latent1;
    		if(symbol1>max_value1)symbol1_clamped=max_value1;
    		else if(symbol1<min_value1)symbol1_clamped=min_value1;
    		else symbol1_clamped=symbol1;
    		symbol.data[i]=symbol1_clamped;


    		latent1=((ap_int<a_bitw+2>)symbol1_clamped<<(a_bitw-8))+param_reg1.data[i*2];
    		if(latent1>MAX_A_VALUE)dequant_reg.data[i]=MAX_A_VALUE;
    		else if(latent1<-MAX_A_VALUE)dequant_reg.data[i]=-MAX_A_VALUE;
    		else dequant_reg.data[i]=latent1;
    		scale_output.data[i]=scale_index1;


    		//			get symbol_2, scale_2,latent_2
    		ap_uint<8> scale_index2;
			if(param_reg2.data[i*2+1]>127)scale_index2=127;
			else if(param_reg2.data[i*2+1]<0)scale_index2=0;
			else scale_index2=(ap_uint<8>)param_reg2.data[i*2+1];
			ap_int<a_bitw+2> symbol2=((ap_int<a_bitw+2>)input.data[i+PCN/2]-param_reg2.data[i*2]+half_value)>>(a_bitw-8);
			ap_int<a_bitw> max_value2=symbol_max_value[scale_index2];
			ap_int<a_bitw> min_value2=symbol_min_value[scale_index2];

			ap_int<a_bitw> symbol2_clamped;
			ap_int<a_bitw+2> latent2;
			if(symbol2>max_value2)symbol2_clamped=max_value2;
			else if(symbol2<min_value2)symbol2_clamped=min_value2;
			else symbol2_clamped=symbol2;
			symbol.data[i+PCN/2]=symbol2_clamped;

			latent2=((ap_int<a_bitw+2>)symbol2_clamped<<(a_bitw-8))+param_reg2.data[2*i];
			if(latent2>MAX_A_VALUE)dequant_reg.data[i+PCN/2]=MAX_A_VALUE;
			else if(latent2<-MAX_A_VALUE)dequant_reg.data[i+PCN/2]=-MAX_A_VALUE;
			else dequant_reg.data[i+PCN/2]=latent2;
			scale_output.data[i+PCN/2]=scale_index2;


    	}
    }

    void quant_output(
    		ap_uint<4> layer_num,
			AggData<a_T,PCN> activation_outbuffer[3*SCM][MAX_W+2],

			hls::stream<AggData<a_T,2*(PCN/2)>> &latent_stream,//bundles of PCN elements contain PCN/2 channels and 2 contiguous spatial locations on horizontal level

			hls::stream<AggData<a_T,PCN>> &add_stream1,
			hls::stream<AggData<a_T,PCN>> &add_stream2,

    		hls::stream<AggData<buf_T,PCN>> &mac_stream1,
			hls::stream<AggData<buf_T,PCN>> &mac_stream2,

			hls::stream<AggData<ap_int<8>,PCN>> &symbol_stream,
			hls::stream<AggData<ap_int<8>,PCN>> &pdf_index_stream){
    	ap_uint<8> offset=0;
    	AggData<ap_int<a_bitw>, PCN> dequant_latent_buffer[MAX_W/2];
     	bool last_layer_flag=(layer_num==2);
    	bool add_flag=(layer_num==1);
    	bool apply_relu_flag=(layer_num!=2);
    	if(layer_num==2)offset=input_line_ptr*SCM;//offset of rotation buffer activation_buffer[2],currently stores row h-3, will be updated with row h


		for(int pocn_index=0;pocn_index<2*SCM;pocn_index++){
			bool apply_relu=(layer_num!=2);
			bool even_channel_flag=((pocn_index%2)==0);// This flag is used to decide whether to cache dequantized latent or concat cached latent with newly dequantized latent to store in activation buffer

			AggData<scale_T,PCN> scale_reg;
			AggData<bias_T,PCN> bias_reg;
			#pragma HLS ARRAY_PARTITION variable=scale_reg.data type=complete
			#pragma HLS ARRAY_PARTITION variable=bias_reg.data type=complete

			scale_reg=scales[scale_bias_index];
			bias_reg=bias[scale_bias_index];
			const ap_uint<16> halved_pocn_index=(pocn_index>>1);

			scale_bias_index=scale_bias_index+1;
			if(scale_bias_index==TOTAL_SOCN_NUM)scale_bias_index=0;
			for(int k=0;k<MAX_W/2;++k){
				#pragma HLS PIPELINE II=3
				AggData<buf_T,PCN> read_reg1;
				AggData<buf_T,PCN> read_reg2;

				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=read_reg1.data
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=read_reg2.data

				read_reg1=mac_stream1.read();
				read_reg2=mac_stream2.read();


				AggData<a_T,PCN> out_reg1;
				AggData<a_T,PCN> out_reg2;
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=out_reg1.data
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=out_reg2.data

				AggData<a_T,PCN> latent_reg;
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=latent_reg.data

				AggData<a_T,PCN> ext_input_reg1;
				AggData<a_T,PCN> ext_input_reg2;
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=ext_input_reg1.data
				#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=ext_input_reg2.data


				if(add_flag){
					ext_input_reg1=add_stream1.read();
					ext_input_reg2=add_stream2.read();
				}

				if(last_layer_flag){
					latent_reg=latent_stream.read();

				}

				agg_bias_scale_quant
				 (out_reg1, read_reg1,scale_reg, bias_reg);
				agg_bias_scale_quant
				 (out_reg2, read_reg2,scale_reg, bias_reg);


				if(add_flag){
					AggADD<a_bitw,PCN>(out_reg1,ext_input_reg1);
					AggADD<a_bitw,PCN>(out_reg2,ext_input_reg2);

				}

				if(apply_relu_flag){
					AggLeakyReLU<a_bitw,PCN>(out_reg1);
					AggLeakyReLU<a_bitw,PCN>(out_reg2);
				}


				if(last_layer_flag){
					AggData<ap_int<8>,PCN> symbol;
					AggData<ap_int<8>,PCN> scale_output;
					AggData<ap_int<a_bitw>,PCN> dequant_reg;
					AggData<ap_int<a_bitw>,PCN> dequant_read_reg;
					#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=symbol.data
					#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=scale_output.data
					#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=dequant_reg.data


					// DATA IN SYMBOL: 2 x (PCN/2), 2 in width, PCN/2 in channel
					if(not even_channel_flag)dequant_read_reg=dequant_latent_buffer[k];

					//latent and symbol bundles of PCN elements contain PCN/2 channels and 2 contiguous spatial locations on horizontal level
					//out_reg1 of PCN elements contains interleaved means and scales, PCN/2 means and PCN/2 scales for the PCN/2 latents
					//out_reg2 of PCN elements contains interleaved means and scales, PCN/2 means and PCN/2 scales for another PCN/2 latents
					QuantLatent(symbol,scale_output,dequant_reg,latent_reg,out_reg1,out_reg2);


					//outputs symbol and corresponding pdf index tile of 2*(PCN/2), PCN/2 channels, 2 in width
					symbol_stream.write(symbol);
					pdf_index_stream.write(scale_output);
					if(even_channel_flag)dequant_latent_buffer[k]=dequant_reg;// when in even channel, store 2*(PCN/2) dequant elements into buffer
					else{	// when in even channel, read 2*(PCN/2) elements from the previous iteration from buffer
							// concate PCN/2 elements from buffer with PCN/2 elements from dequant_reg to form a full PCN(channel dimension) bundle.

							AggData<a_T,PCN> out_reg1;
							AggData<a_T,PCN> out_reg2;
							#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=out_reg1.data
							#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=out_reg2.data
							reshape_to_output_loop:
							for(int i=0;i<PCN/2;i++){
								#pragma HLS UNROLL
								out_reg1.data[i]=dequant_read_reg.data[i];
								out_reg1.data[i+PCN/2]=dequant_reg.data[i];

								out_reg2.data[i]=dequant_read_reg.data[i+PCN/2];
								out_reg2.data[i+PCN/2]=dequant_reg.data[i+PCN/2];
							}
							//Updating the buffer cells current storing row h-3 with row h
							activation_outbuffer[halved_pocn_index+offset][2*k+1]=out_reg1;
							activation_outbuffer[halved_pocn_index+offset][2*k+2]=out_reg2;
					}
				}
				else{
					activation_outbuffer[pocn_index+offset][2*k+1]=out_reg1;
					activation_outbuffer[pocn_index+offset][2*k+2]=out_reg2;
				}
			}
		}
		if (layer_num==2){
			if(input_line_ptr==2)input_line_ptr=0;
			else input_line_ptr=input_line_ptr+1;
		}
	}




    void input_feeder(ap_uint<4> layer_num,
    		AggData<a_T,PCN> a_buffer[2*SCM][MAX_W+2],
    		hls::stream<AggData<a_T,PCN>> &act_stream1,
			hls::stream<AggData<a_T,PCN>> &act_stream2){
    		const ap_uint<8> cur_block_num=block_num[layer_num];

    		ap_uint<16> start_channel_index=0;//channel pointer offset,for layer1,2 start_channel_index remains 0
    		//for layer 0, start_channel_index=first_layer_start_channel_index point to the top line(oldest line) in rotaion buffer activation_buffer[2]
    		if(layer_num==0)start_channel_index=first_layer_start_channel_index;
    		else if(layer_num==1){
    			// update first_layer_start_channel_index in advance for the layer 0 access in the iteration of the next line
    			if(first_layer_start_channel_index==2*SCM)first_layer_start_channel_index=0;
    			else first_layer_start_channel_index+=SCM;
    		}
			for(ap_uint<16> odepth=0;odepth<2*SCM;odepth++){
				for(ap_uint<16> block_index=0;block_index<cur_block_num;++block_index){
					#pragma HLS LOOP_TRIPCOUNT min=9 max=27 avg=16
					ap_uint<16> c_index=channel_ptr[block_ptr_idx]+start_channel_index;//get the channel index for current block of weights
					if (c_index>=3*SCM) c_index-=3*SCM;// only applies for layer 0, implemets the rotation buffer which points to the correct data in activation_buffer[2]

					ap_uint<4> h_index=horizontal_offset[block_ptr_idx];//get the horizontal location of the current block inside the convolution kernel.

					if(block_ptr_idx==WB_NUM-1)block_ptr_idx=0;
					else block_ptr_idx++;

					for(int k=0;k<MAX_W/2;++k){
						#pragma HLS PIPELINE II=1
						AggData<a_T,PCN> act_reg1;
						AggData<a_T,PCN> act_reg2;

						#pragma HLS AGGREGATE compact=auto variable=act_reg1
						#pragma HLS AGGREGATE compact=auto variable=act_reg2


						act_reg1=a_buffer[c_index][k*2+h_index];
						act_reg2=a_buffer[c_index][k*2+h_index+1];

						act_stream1.write(act_reg1);
						act_stream2.write(act_reg2);
					}
				}
			}

    }

    void run_one_layer(ap_uint<4> layer_num,
        		AggData<a_T,PCN> input_act_buffer[3*SCM][MAX_W+2],
    			AggData<a_T,PCN> output_act_buffer[3*SCM][MAX_W+2],
				hls::stream<AggData<a_T,PCN>> &latent_stream,

				hls::stream<AggData<a_T,PCN>> &add_stream1,
				hls::stream<AggData<a_T,PCN>> &add_stream2,
				hls::stream<AggData<ap_int<8>,PCN>> &symbol_stream,
				hls::stream<AggData<ap_int<8>,PCN>> &pdf_index_stream){

		#pragma HLS INLINE off
		#pragma HLS INTERFACE mode=bram port=output_act_buffer
		#pragma HLS INTERFACE mode=bram port=input_act_buffer
    	hls::stream<AggData<a_T,PCN>> act_stream1;
    	hls::stream<AggData<a_T,PCN>> act_stream2;


    	hls::stream<AggData<buf_T,PCN>> mac_stream1;
    	hls::stream<AggData<buf_T,PCN>> mac_stream2;


		#pragma HLS DATAFLOW

		input_feeder(layer_num,input_act_buffer,
			act_stream1,act_stream2);

		run_mac(layer_num,
				act_stream1,act_stream2,
				mac_stream1,mac_stream2);

		quant_output(layer_num,output_act_buffer,
					latent_stream,
					add_stream1,add_stream2,
					mac_stream1,mac_stream2,
					symbol_stream,pdf_index_stream);

    }

    void run(   ap_uint<16> height,
				hls::stream<AggData<a_T,2*(PCN/2)>> &latent_stream, //bundles of PCN elements contain PCN/2 channels and 2 contiguous spatial locations on horizontal level
				hls::stream<AggData<a_T,PCN>> &add_stream1,
				hls::stream<AggData<a_T,PCN>> &add_stream2,
				hls::stream<AggData<ap_int<8>,PCN>> &symbol_stream,
				hls::stream<AggData<ap_int<8>,PCN>> &pdf_index_stream)
    {
        #pragma HLS STREAM variable=activation_buffer off
    	input_line_ptr=0;
    	block_ptr_idx=0;
    	AggData<a_T,PCN>act_zero_reg;
    	AggInit<a_T,PCN>(act_zero_reg);
    	wb_index=0;
		scale_bias_index=0;
		weight_pingpong=0;
		first_layer_start_channel_index=0;
		load_weight_reg(weight_reg[weight_pingpong]);

    	pad_three_lines_loop:
    	for(int j=0;j<3*SCM;j++){
    		for(int k=0;k<(MAX_W+2);k++){
    			activation_buffer[2][j][k]=act_zero_reg;
    		}
    	}

    	ape_height_loop:
        for(ap_uint<16> i = 0; i < height; i++) {
    		#pragma HLS LOOP_TRIPCOUNT min=1 max=45
//        	layer 0,1,2 are performed sequetially, without overlapping
    		run_one_layer(0,
			  activation_buffer[2], activation_buffer[0],
			  latent_stream,
			  add_stream1, add_stream2,
			  symbol_stream,pdf_index_stream);
    		run_one_layer(1,
			  activation_buffer[0], activation_buffer[1],
			  latent_stream,
			  add_stream1, add_stream2,
			  symbol_stream,pdf_index_stream);
    		run_one_layer(2,
			  activation_buffer[1], activation_buffer[2],
			  latent_stream,
			  add_stream1, add_stream2,
			  symbol_stream,pdf_index_stream);
    	}
    }
};

#undef w_T
#undef w_wg31
#undef a_T
#undef out_T
#undef buf_T

#endif
