#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "common.hpp"


template<typename D_T,int PCN,int SCN,int MAX_W>
void CatHorizontal(ap_uint<16> height,// height is the height before pixel shuffle, MAX_W is also before pixel shuffle
		hls::stream<AggData<D_T,PCN>> &in_stream,
		hls::stream<AggData<D_T,2*PCN>> &out_stream){
	height_loop:
	for(ap_uint<16> i=0;i<height/2;++i){
		channel_loop:
		for(ap_uint<10> j=0;j<SCN;++j){
		width_loop:
			for(ap_uint<16> k=0;k<MAX_W/4;++k){
				#pragma HLS PIPELINE II=2
				AggData<D_T,PCN> in_reg1;
				AggData<D_T,PCN> in_reg2;
				AggData<D_T,2*PCN> out_reg;
				#pragma HLS AGGREGATE compact=auto variable=in_reg1
				#pragma HLS AGGREGATE compact=auto variable=in_reg2
				#pragma HLS AGGREGATE compact=auto variable=out_reg
				in_reg1=in_stream.read();
				in_reg2=in_stream.read();
				CatAgg<D_T,PCN>(out_reg,in_reg1,in_reg2);
				out_stream.write(out_reg);
			}
		}
		std::cout<<std::endl;
	}

}



//
template<typename D_T,int PCN,int SCN,int MAX_W>//MAX_W is after pixel un_shuffle
void pad_split_stream22(ap_uint<16> height,
		hls::stream<AggData<D_T,8*PCN>> &in_stream,
		hls::stream<AggData<D_T,8*PCN>> &out_stream0,
		hls::stream<AggData<D_T,8*PCN>> &out_stream1){
	bool to_stream1_flag=true;

	AggData<D_T,8*PCN> in_reg;
	#pragma HLS ARRAY_PARTITION variable=in_reg.data type=complete

	AggData<D_T,PCN> line_buffer[SCN][MAX_W*2];
	#pragma HLS AGGREGATE variable=line_buffer compact=auto
	#pragma HLS ARRAY_PARTITION dim=2 type=cyclic factor=2 variable=line_buffer


	AggData<D_T,8*PCN> write_reg;
	#pragma HLS ARRAY_PARTITION variable=write_reg.data type=complete

	AggData<D_T,PCN> shift_reg0;
	#pragma HLS AGGREGATE variable= shift_reg0
	AggData<D_T,PCN> shift_reg1;
	#pragma HLS AGGREGATE variable= shift_reg1



	AggData<D_T,PCN> in_array[8];
	#pragma HLS ARRAY_PARTITION variable=in_array dim=1 type=complete
	#pragma HLS AGGREGATE variable=in_array compact=auto


	AggData<D_T,PCN> write_array[8];
	#pragma HLS ARRAY_PARTITION variable=write_array dim=1 type=complete
	#pragma HLS AGGREGATE variable=write_array compact=auto

	AggData<D_T,PCN> zero_reg;
	#pragma HLS ARRAY_PARTITION variable=zero_reg.data type=complete
	AggInit<D_T,PCN>(zero_reg);

	for(ap_uint<16> j=0;j<SCN;j++){
		for(ap_uint<16> k=0;k<=MAX_W/2;++k){
			line_buffer[j][k*4]=zero_reg;
			line_buffer[j][k*4+1]=zero_reg;
			line_buffer[j][k*4+2]=zero_reg;
			line_buffer[j][k*4+3]=zero_reg;
		}
	}

	//pad 1 on the top,1 on the bottom(before unshuffle), which means one more line after unshuffle.
	height_loop:
	for(ap_uint<16> i=0;i<=height;++i){
		bool read_flag=(i!=height);
		channel_loop:
		for(ap_uint<10> j=0;j<SCN;++j){
			shift_reg0=zero_reg;
			shift_reg1=zero_reg;
        // input vector: 2x4(before unshuffle)
		width_loop:
			for(ap_uint<16> k=0;k<=MAX_W/2;++k){
				#pragma HLS PIPELINE II=4
				write_array[0]=shift_reg0;
				write_array[2]=shift_reg1;

				if(k!=MAX_W/2){
					if (read_flag)in_reg=in_stream.read();
					SplitNAgg<D_T,PCN,8>(in_reg,in_array);
					write_array[1]=line_buffer[j][k*4];
					write_array[4]=line_buffer[j][k*4+1];
					write_array[5]=line_buffer[j][k*4+2];
					if(read_flag){
						write_array[3]=in_array[0];
						write_array[6]=in_array[1];
						write_array[7]=in_array[4];
					}
					else{
						write_array[3]=zero_reg;
						write_array[6]=zero_reg;
						write_array[7]=zero_reg;
					}
				}
				else{
					write_array[1]=zero_reg;
					write_array[4]=zero_reg;
					write_array[5]=zero_reg;
					write_array[3]=zero_reg;
					write_array[6]=zero_reg;
					write_array[7]=zero_reg;
				}

				CatNAgg<D_T,PCN,8>(write_reg,write_array);
				if(to_stream1_flag)out_stream1.write(write_reg);
				else out_stream0.write(write_reg);
//				if(j==0&&!(to_stream1_flag))std::cout<<write_reg.data[0]<<','<<write_reg.data[PCN]<<','<<write_reg.data[4*PCN]<<','<<write_reg.data[5*PCN]<<',';


				if(k!=MAX_W/2){
					shift_reg0=line_buffer[j][k*4+3];
					if(read_flag){
						shift_reg1=in_array[5];
						line_buffer[j][k*4]=in_array[2];
						line_buffer[j][k*4+1]=in_array[3];
						line_buffer[j][k*4+2]=in_array[6];
						line_buffer[j][k*4+3]=in_array[7];
					}
					else{
						shift_reg1=zero_reg;
					}
				}
			}
		}
		to_stream1_flag=!to_stream1_flag;
	}
}




template<typename D_T,int PCN>
void copy_array(D_T output[PCN],D_T input[PCN]){
	for(int i=0;i<PCN;i++){
		#pragma HLS UNROLL
		output[i]=input[i];
	}
}



#endif
