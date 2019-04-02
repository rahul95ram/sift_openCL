__constant float4 grayscale = { 0.2989f, 0.5870f, 0.1140f, 0 };

__kernel void DownSample (  __read_only image2d_t inputImg,
                            __write_only image2d_t outputImg,
							int SIZEX, int SIZEY, 
							sampler_t sampler)
{
 
    int i = get_global_id (0);//index of the output image
    int j = get_global_id (1);
	int row = get_global_size(0)*SIZEX;//size of the input image
	int col = get_global_size(1)*SIZEY;
    // jump to the starting indexes
    int is = i * SIZEX;
    int js = j * SIZEY;
    //printf("(%d,%d)\n",i,j);

   float4 total = (float4) (0);
  
   for ( int x = 0; x < SIZEX; x++ ) {
        for ( int y = 0; y < SIZEY; y++ ) {
                total += read_imagef ( inputImg, sampler, (int2) ( is + x, js + y ) );
        }
    }

    total = (float4) ( total / (SIZEX*SIZEY) );
	
    write_imagef ( outputImg, (int2) ( i, j ), total );
}

__kernel void GaussianFilter(int filterWidth, 
                             __global float* sigma,
							 __global float * gaussBlurFilter,
							 __global float * filtSum)
{
	float gauss[10];
	for(int i=0;i<filterWidth;i++)
	{
		gauss[i]=-1+i*(float)2/(filterWidth-1);
		
	}
	int filtIdx = get_local_id(0);
	int filtSize = get_local_size(0);
	int x = filtIdx%filterWidth;
	int y = filtIdx/filterWidth;
	int z = get_global_id(1);
	//printf("%d\n",z*filtSize+filtIdx);
	gaussBlurFilter[z*filtSize+filtIdx] = (float)1/(2*3.14159*sigma[z])*exp((-gauss[x]*gauss[x]-gauss[y]*gauss[y])/(2*sigma[z]*sigma[z]));
	filtSum[z*filtSize+filtIdx] = work_group_scan_inclusive_add(gaussBlurFilter[z*filtSize+filtIdx]);
	gaussBlurFilter[z*filtSize+filtIdx]=(float)gaussBlurFilter[z*filtSize+filtIdx]/filtSum[z*filtSize+filtSize-1];
	//printf(" After %d, %f\n",z,filtSum[z*filtSize+filtSize-1]);

}

__kernel void GaussianBlur(__read_only image2d_t inputImg, 
                           __write_only image2d_array_t outputImg,
						   sampler_t sampler, int filterWidth,
						   __global float *gaussBlurFilter)
{

	// use global IDs for output coords
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // sigmas

	int halfWidth = (int)(filterWidth/2); // auto-round nearest int ???
	int filtSize = filterWidth*filterWidth;
	float4 sum = (float4)(0);
	int filtIdx = 0; // filter kernel passed in as linearized buffer array
	int2 coords;
	for(int i = -halfWidth; i <= halfWidth; i++) // iterate filter rows
	{
		coords.y = y + i;
		for(int j = -halfWidth; j <= halfWidth; j++) // iterate filter cols
	  {
	  coords.x = x + j;
	  //float4 pixel = convert_float4(read_imageui(inputImg, sampler, coords)); // operate element-wise on all 3 color components (r,g,b)
	  float4 pixel = read_imagef(inputImg, sampler, coords); // operate element-wise on all 3 color components (r,g,b)
	  filtIdx++;
	  sum += pixel * (float4)(gaussBlurFilter[z*filtSize+filtIdx],gaussBlurFilter[z*filtSize+filtIdx],gaussBlurFilter[z*filtSize+filtIdx],1.0f); // leave a-channel unchanged
	  //printf("sum : %f,%f,%f,%f\n",sum.x,sum.y,sum.z,sum.w);
	  }
     }
	//write resultant filtered pixel to output image
	int4 coords2 = (int4)(x,y,z,0);
	//write_imageui(outputImg, coords, convert_uint4(sum));
	write_imagef(outputImg, coords2, sum);
}


__kernel void DifferenceOfGaussian(__read_only image2d_array_t inputImg, __write_only image2d_array_t outputImg, sampler_t sampler)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int4 coords_dog = (int4)(x,y,z,0);
	int4 coords = (int4)(x,y,z+1,0);

	float4 pixel1 = read_imagef(inputImg, sampler, coords_dog);
	float4 pixel2 = read_imagef(inputImg, sampler, coords);


	float4 DoG = pixel1 - pixel2;
	write_imagef(outputImg, coords_dog, DoG);

}

__kernel void Extrema(__read_only image2d_array_t DoGImg,
					  __global int * extrema,
					  sampler_t sampler,
					  __global int *index)
{ 
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // index of image
	int w = get_global_size(0);
	int h = get_global_size(1);

	float4 rgba;
	int idx = 0;
	__local float p[27];
	if(x!=0 && x!=(w-1) && y!=0 && y!=(h-1))
	{
		for(int i=0;i<3;i++)
		{
			for(int j=-1;j<=1;j++)
			{ 
				for(int k=-1;k<=1;k++)
				{
					rgba = read_imagef(DoGImg,sampler,(int4)(x+j,y+k,z+i,0));
					//printf("%f\n",rgba.x);
					p[i*9+(j+1)*3+(k+1)] = (float) dot(grayscale, rgba);
				}			   
			}		
		}
		
		if(p[13]>0)
		{
		    //printf("p[13]:%f\n",p[13]);
			if(p[13]>p[0] && p[13]>p[1] && p[13]>p[2] && p[13]>p[3] && p[13]>p[4] && p[13]>p[5]
			&& p[13]>p[6] && p[13]>p[7] && p[13]>p[8] && p[13]>p[9] && p[13]>p[10] && p[13]>p[11]
			&& p[13]>p[12] && p[13]>p[14] && p[13]>p[15] && p[13]>p[16] && p[13]>p[17] && p[13]>p[18]
			&& p[13]>p[19] && p[13]>p[20] && p[13]>p[21] && p[13]>p[22] && p[13]>p[23] && p[13]>p[24]
			&& p[13]>p[25] && p[13]>p[26])
			{
		
				idx = atomic_add(index,1);
				printf("(%d,%d,%d)\n",x,y,z+1);
				extrema[idx*3]=x;
				extrema[idx*3+1]=y;
				extrema[idx*3+2]=z+1;
			}	   
		}
		else if(p[13]<0)
		{
			if(p[13]<p[0] && p[13]<p[1] && p[13]<p[2] && p[13]<p[3] && p[13]<p[4] && p[13]<p[5]
			&& p[13]<p[6] && p[13]<p[7] && p[13]<p[8] && p[13]<p[9] && p[13]<p[10] && p[13]<p[11]
			&& p[13]<p[12] && p[13]<p[14] && p[13]<p[15] && p[13]<p[16] && p[13]<p[17] && p[13]<p[18]
			&& p[13]<p[19] && p[13]<p[20] && p[13]<p[21] && p[13]<p[22] && p[13]<p[23] && p[13]<p[24]
			&& p[13]<p[25] && p[13]<p[26])
			{
		
				idx = atomic_add(index,1);
				printf("(%d,%d,%d)\n",x,y,z+1);
				extrema[idx*3]=x;
				extrema[idx*3+1]=y;
				extrema[idx*3+2]=z+1;
			}	   
		}	
		
	}
	
}
//Change gloabl work size to make sure edge cases are taken care of

					  
__kernel void KeyPoints(__read_only image2d_array_t DoGImg,
					    __global uint4 * extrema, 
					    __global uint4 * keypoints,
					    sampler_t sampler,
					    __global int *index)
{ 

		int i = get_global_id(0);
		int x = extrema[i].x;
		int y = extrema[i].y;
		int z = extrema[i].z;
		
		//get the derivative of dD/dx
		int4 coords1,coords2;
		coords1 = (int4)(x-1,y,z,0); coords2 = (int4)(x+1,y,z,0);
		float4 D0 = (read_imagef(DoGImg, sampler, coords2)-read_imagef(DoGImg, sampler, coords1)) / 2.0f;//x
		coords1 = (int4)(x,y-1,z,0); coords2 = (int4)(x,y+1,z,0);
		float4 D1 = (read_imagef(DoGImg, sampler, coords2)-read_imagef(DoGImg, sampler, coords1)) / 2.0f;//y
		coords1 = (int4)(x,y,z-1,0); coords2 = (int4)(x,y,z+1,0);
		float4 D2 = (read_imagef(DoGImg, sampler, coords2)-read_imagef(DoGImg, sampler, coords1)) / 2.0f;//sigma
	
		//get the Hessian Matrix
		int idx =0;
		float h = 1;
		float k = 1;
        
        float4 dxx = (read_imagef(DoGImg, sampler, (int4)(x+h,y,z,0)) - read_imagef(DoGImg, sampler, (int4)(x-h,y,z,0)) - 2*read_imagef(DoGImg, sampler, (int4)(x,y,z,0)) )/(h*h);
			
        float4 dyy = ( read_imagef(DoGImg, sampler, (int4)(x,y+h,z,0))- read_imagef(DoGImg, sampler, (int4)(x,y-h,z,0)) - 2*read_imagef(DoGImg, sampler, (int4)(x,y,z,0)) )/(h*h); 
			
        float4 dss = ( read_imagef(DoGImg, sampler, (int4)(x,y,z+h,0))- read_imagef(DoGImg, sampler, (int4)(x,y,z-h,0)) - 2*read_imagef(DoGImg, sampler, (int4)(x,y,z,0)) )/(h*h); 
			
        float4 dxy = ( read_imagef(DoGImg, sampler, (int4)(x+h,y+k,z,0))- read_imagef(DoGImg, sampler, (int4)(x-h,y-k,z,0)) - read_imagef(DoGImg, sampler, (int4)(x+h,y-k,z,0)) - read_imagef(DoGImg, sampler, (int4)(x-h,y+k,z,0)) )/(4*h*k); 
			
        float4 dxs = ( read_imagef(DoGImg, sampler, (int4)(x+h,y,z+k,0))- read_imagef(DoGImg, sampler, (int4)(x-h,y,z-k,0)) - read_imagef(DoGImg, sampler, (int4)(x+h,y,z-k,0)) - read_imagef(DoGImg, sampler, (int4)(x-h,y,z+k,0)) )/(4*h*k);
			
        float4 dys = ( read_imagef(DoGImg, sampler, (int4)(x,y+k,z+h,0))- read_imagef(DoGImg, sampler, (int4)(x,y-k,z-h,0)) - read_imagef(DoGImg, sampler, (int4)(x,y-k,z+h,0)) - read_imagef(DoGImg, sampler, (int4)(x,y+k,z-h,0)) )/(4*h*k);		

        float4 H00 = dxx; //H00
        float4 H01 = dxy; //H01
        float4 H02 = dxs; //H02
        float4 H10 = dxy; //H10
        float4 H11 = dyy; //H11
        float4 H12 = dys; //H12
        float4 H20 = dxs; //H20
        float4 H21 = dys; //H21
        float4 H22 = dss; //H22

		//inversion of the Hessian

		float4 det = -(H02*H11*H20) + H01*H12*H20 + H02*H10*H21 - H00*H12*H21 - H01*H10*H22 + H00*H11*H22;
		//printf(" Det - %v4f", det);
		float4 K00 = H11*H22 - H12*H21;
		float4 K01 = H02*H21 - H01*H22;
		float4 K02 = H01*H12 - H02*H11;
		float4 K10 = H12*H20 - H10*H22;
		float4 K11 = H00*H22 - H02*H20;
		float4 K12 = H02*H10 - H00*H12;
		float4 K20 = H10*H21 - H11*H20;
		float4 K21 = H01*H20 - H00*H21;
		float4 K22 = H00*H11 - H01*H10;

		//x = -H^(-1)*D
		float4 solution0 = -(D0*K00 + D1*K01 + D2*K02)/det; //x
		float4 solution1 = -(D0*K10 + D1*K11 + D2*K12)/det; //y
		float4 solution2 = -(D0*K20 + D1*K21 + D2*K22)/det; //sigma

		//interpolated DoG magnitude at this peak
		int4 coords = (int4)(x,y,z,0);
		float4 Dx = read_imagef(DoGImg, sampler, coords) + 0.5f * (solution0*D0+solution1*D1+solution2*D2);

		float4 TrH = dxx+dyy+dss;
		// use two threshold for selectint the key points
		float4 Tr_D = TrH*TrH/det;
		//printf("(%f,%f,%f,%f)",Tr_D.x,Tr_D.y,Tr_D.z,Tr_D.w);
	    if(Dx.x>0.03 && Dx.y>0.03 && Dx.z>0.03 && Tr_D.x>12.1 && Tr_D.y>12.1 && Tr_D.z>12.1) //discard the extrema with |D(x)|<0.03
		{ 
			idx = atomic_add(index,1);
			printf("(%d,%d,%d)\n",x,y,z);
			keypoints[idx]=(uint4)(x,y,z,0);
		}
		
}





				

			

