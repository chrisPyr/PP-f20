#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  
  __pp_vec_int y;
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float test=_pp_vset_float(9.999999f);
  __pp_vec_int count;
  //__pp_vec_float zero=_pp_vset_float(0.f);
  __pp_vec_float ones=_pp_vset_float(1.0f);
  __pp_vec_int izero=_pp_vset_int(0);
  __pp_vec_int ione=_pp_vset_int(1);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;  
  __pp_mask maskIsGL,maskIszero=_pp_init_ones(0); 
  maskAll = _pp_init_ones();
 
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
	int t=0;
	int k=i;
	//count how many values in this vector are valid
	while(++k<=N && t<VECTOR_WIDTH){
		++t;
	}

	//initialize mask for valid values
	maskAll = _pp_init_ones(t);
	//load  values from  valid positions
	_pp_vload_float(x, values+i, maskAll);
	_pp_vload_int(y,exponents+i,maskAll);
	//initialize result
	_pp_vload_float(result, values+i,maskAll);
	//all zeros
	maskIsNegative = _pp_init_ones(0);
	//all ones
	maskIsNotNegative= _pp_mask_not(maskIsNegative);
	//exclude the invalid positions
	maskIsNotNegative = _pp_mask_and(maskIsNotNegative,maskAll);
	maskIsGL = _pp_mask_and(maskIsGL,maskAll);
	maskIszero = _pp_mask_and(maskIszero,maskAll);
	//test if the exponent is zero
	_pp_veq_int(maskIszero,y,izero,maskAll);
	
	//do multiply until all exponents decreases to zero	
	while(_pp_cntbits(maskIsNotNegative)!=0){

	_pp_vsub_int(y,y,ione,maskAll);
	_pp_vlt_int(maskIsNegative,y,ione,maskAll);	
	maskIsNotNegative=_pp_mask_not(maskIsNegative);
	maskIsNotNegative =_pp_mask_and (maskIsNotNegative,maskAll);
	_pp_vmult_float(result,result,x,maskIsNotNegative);

	}
	
	//test  the value from which position  is larger than 9.999999
	_pp_vgt_float(maskIsGL,result,test,maskAll);
	//store the result
	_pp_vstore_float(output+i,result,maskAll);
	//overwrite the result to 9.999999 by the maskIsGL(which contains the positions of values larger than 9.999999)
	_pp_vstore_float(output+i,test,maskIsGL);
	//overwrite the result to 1.0 by the maskIszero(which contains the positions of values of 0)
	_pp_vstore_float(output+i,ones,maskIszero);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  int y=VECTOR_WIDTH;
  float ans;
  __pp_vec_float result= _pp_vset_float(0.0f);
  __pp_vec_float x;
  __pp_mask maskAll = _pp_init_ones();
  __pp_mask maskans = _pp_init_ones(1);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
	_pp_vload_float(x,values+i,maskAll);
	_pp_vadd_float(result,result,x,maskAll);
  }
  while(y/2>1){
  _pp_hadd_float(result,result);
  _pp_interleave_float(result,result);
  y/=2;
  } 
  _pp_hadd_float(result,result);
  _pp_vstore_float(&ans,result,maskans);
  return ans;
}
