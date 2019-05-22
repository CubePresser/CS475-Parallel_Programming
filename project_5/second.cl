kernel
void
ArrayMultReduce( global const float *dA, global const float *dB, local float *prods, global float *dC )
{
	int gid = get_global_id( 0 );
	int numitems = get_local_size( 0 );
	int tnum = get_local_id( 0 );
	int wgnum = get_group_id( 0 );

	// Get the products
	prods[tnum] = dA[gid] * dB[gid];

	// Add the products up
	for(int offset = 1; offset < numitems; offset *= 2) {
		int mask = 2*offset - 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if( (tnum & mask) == 0 ) {
			prods[tnum] += prods[tnum + offset];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if(tnum == 0) {
		dC[wgnum] = prods[0];
	}
}