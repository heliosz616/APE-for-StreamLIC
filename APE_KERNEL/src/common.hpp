#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <cassert>
#include "ap_int.h"
#include<iostream>
#include "hls_stream.h"

// types for inference
typedef ap_uint<6> uint_s;
typedef ap_uint<12> uint_p;
typedef ap_uint<32> apu_32;
typedef ap_int<12> fix_k;
typedef ap_int<40> fix_m;
typedef ap_int<16> int_scale;
typedef ap_int<16> int_offset;
typedef ap_int<16> bias_T;
typedef ap_int<32> ex_bias_T;
typedef ap_int<48> int_48;

//types for entropy coding
const int STATE_BITS = 32;
const int TOTAL_FREQ_BITS = 16;
const int NUM_SYMBOLS = 256;

typedef ap_uint<STATE_BITS> state_T;
typedef ap_uint<TOTAL_FREQ_BITS+1> freq_T;
typedef ap_uint<64> prd_T;
typedef ap_uint<8> symbol_T;

#define S1 1


struct scale_T{
	ap_uint<16> param;
	ap_uint<6> shift;
};

template<typename D_T,unsigned short C_N>
struct AggData{
	D_T data[C_N];

};



template<typename out_T,typename in_T,unsigned short C_N>
void AggConvert(AggData<out_T,C_N> &out,AggData<in_T,C_N> in){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		out.data[i]=(out_T)in.data[i];
	}
};

template<typename D_T,unsigned short C_N>
void Agg2Array(AggData<D_T,C_N> agg,D_T a[C_N]){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		a[i]=agg.data[i];
	}
};

template<typename D_T,unsigned short C_N>
void Agg2twoArray(AggData<D_T,2*C_N> agg,D_T a[C_N],D_T b[C_N]){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		a[i]=agg.data[i];
		b[i]=agg.data[C_N+i];
	}
};

template<typename D_T,unsigned short C_N>
void Array2Agg(AggData<D_T,C_N> &agg,D_T a[C_N]){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		agg.data[i]=a[i];
	}
};

template<typename D_T,unsigned short C_N>
void CatAgg(AggData<D_T,2*C_N> &out,AggData<D_T,C_N> a,AggData<D_T,C_N> b){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		out.data[i]=a.data[i];
		out.data[C_N+i]=b.data[i];
	}
};

template<typename D_T,unsigned short C_N,unsigned short N>
void CatNAgg(AggData<D_T,N*C_N> &out,AggData<D_T,C_N> a[N]){
#pragma HLS INLINE
	for(int i=0;i<N;i++){
		#pragma HLS UNROLL
		for(int j=0;j<C_N;j++)
		{
			out.data[i*C_N+j]=a[i].data[j];
		}

	}
};

template<typename D_T,unsigned short C_N>
void SplitAgg(AggData<D_T,2*C_N> in,AggData<D_T,C_N> &a,AggData<D_T,C_N> &b){
#pragma HLS INLINE
	for(int i=0;i<C_N;i++){
		#pragma HLS UNROLL
		a.data[i]=in.data[i];
		b.data[i]=in.data[i+C_N];
	}
};

template<typename D_T,unsigned short C_N,unsigned short N>
void SplitNAgg(AggData<D_T,N*C_N> in,AggData<D_T,C_N> out[N]){
	for(int i=0;i<N;i++){
		#pragma HLS UNROLL
		for(int j=0;j<C_N;j++){
			#pragma HLS UNROLL
			out[i].data[j]=in.data[C_N*i+j];
		}
	}
};




template<typename D_T,int AGG>
void AggInit(AggData<D_T,AGG> &ptData){
	agg_zero_loop:
	for(int i=0;i<AGG;i++)
	{
		#pragma HLS UNROLL
		ptData.data[i]=0;
	}
}



template<int a_bitw,int AGG>
void AggLeakyReLU(AggData<ap_int<a_bitw>,AGG> &out_reg){
	#pragma HLS INLINE
	for(int i=0;i<AGG;i++){
		#pragma HLS UNROLL
		if(out_reg.data[i]<0)out_reg.data[i]=(ap_int<a_bitw>)(((ap_int<a_bitw+4>)out_reg.data[i]+32)>>6);
	}
}



template<int bitw,int AGG>
void AggADD(AggData<ap_int<bitw>,AGG> &a,AggData<ap_int<bitw>,AGG> b){
	#pragma HLS INLINE
	const ap_int<bitw+2> MAX_V=(1<<(bitw-1))-1;
	agg_add_loop:
	for(int i=0;i<AGG;i++)
	{
		#pragma HLS UNROLL
		ap_int<bitw+2> reg=a.data[i]+b.data[i];
		if(reg>MAX_V)a.data[i]=MAX_V;
		else if(reg<-MAX_V)a.data[i]=-MAX_V;
		else a.data[i]=reg;
	}
}


//template<typename D_T>
//struct exc_T{
//	D_T data;
//	bool flag;
//};

#endif
