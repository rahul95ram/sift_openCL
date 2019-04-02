__constant float4 grayscale = { 0.299f, 0.587f, 0.114f, 0 }; //formula

__kernel void DownSample (  __global unsigned char* inputImg,
                            __global unsigned char* outputImg,
							int SIZEX, int SIZEY)
{
 
    int i = get_global_id (0);//index of the output image
    int j = get_global_id (1);
	int row = get_global_size(0)*SIZEX;//size of the input image
	int col = get_global_size(1)*SIZEY;
    // jump to the starting indexes
    int is = i * SIZEX;
    int js = j * SIZEY;
    //printf("(%d,%d)\n",i,j);

   float total=0.0;
  
   for ( int x = 0; x < SIZEX; x++ ) {
        for ( int y = 0; y < SIZEY; y++ ) {
                total += (float)inputImg[(is+x)*col+(js+y)];
				//printf("total:%f",total);
        }
    }

    total = (float) ( total / (SIZEX*SIZEY) );
	
    outputImg[i*get_global_size(1)+j] = (unsigned char)total;
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

__kernel void GaussianBlur(__global unsigned char* inputImg, 
                           __global unsigned char* outputImg,
						    int filterWidth,
						   __global float *gaussBlurFilter)
{

	// use global IDs for output coords
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // sigmas
	int p = get_global_size(0);
	int s = get_global_size(1);
	
	int halfWidth = (int)(filterWidth/2); // auto-round nearest int ???
	int filtSize = filterWidth*filterWidth;
	float sum = 0.0;
	int filtIdx = 0; // filter kernel passed in as linearized buffer array
	int2 coords;
	for(int i = -halfWidth; i <= halfWidth; i++) // iterate filter rows
	{
		coords.y = y + i;
		for(int j = -halfWidth; j <= halfWidth; j++) // iterate filter cols
	  {
		coords.x = x + j;
		//float4 pixel = convert_float4(read_imageui(inputImg, sampler, coords)); // operate element-wise on all 3 color components (r,g,b)
		float pixel = (float)inputImg[coords.x*s+coords.y]; // operate element-wise on all 3 color components (r,g,b)
		filtIdx++;
		sum += pixel * gaussBlurFilter[z*filtSize+filtIdx]; 
		//printf("sum : %f\n",sum);
	  }
     }
	outputImg[z*p*s+x*s+y]=(unsigned char)sum;
}

__kernel void DifferenceOfGaussian(__global unsigned char* inputImg, 
                                   __global unsigned char* outputImg)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int p = get_global_size(0);
	int s = get_global_size(1);

	float pixel1 = (float)inputImg[(z+1)*p*s+x*s+y];
	float pixel2 = (float)inputImg[z*p*s+x*s+y];

	float DoG = (float)(pixel1 - pixel2);
	//printf("%f\n",DoG);
	outputImg[z*p*s+x*s+y]=(unsigned char) DoG;

}

__kernel void Extrema(__global unsigned char* DoGImg,
					  __global int * extrema,
					  __global int *index)
{ 
   
	int x = get_global_id(0); // cols
	int y = get_global_id(1); // rows
	int z = get_global_id(2); // index of image
	
	int r = get_global_size(0);
	int s = get_global_size(1);
	int idx = 0;
	__local float p[27];
	if(x!=0&&x!=511&&y!=0&&y!=511)
	{
		for(int i=0;i<3;i++)
		{
			for(int j=-1;j<=1;j++)
			{ 
				for(int k=-1;k<=1;k++)
				{
					p[i*9+(j+1)*3+(k+1)] = (float)DoGImg[(z+i)*r*s+(x+j)*s+y+k];
					//printf("%f\n",p[i*9+(j+1)*3+(k+1)]);
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
				//printf("(%d,%d,%d)\n",x,y,z);
				extrema[idx*3]=x;
				extrema[idx*3+1]=y;
				extrema[idx*3+2]=z;
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
				//printf("(%d,%d,%d)\n",x,y,z);
				extrema[idx*3]=x;
				extrema[idx*3+1]=y;
				extrema[idx*3+2]=z;
			}	   
		}	
		
	}
	
}
//Change gloabl work size to make sure edge cases are taken care of

					  
__kernel void KeyPoints(__global unsigned char* DoGImg,
					    __global int * extrema, 
					    __global int * keypoints,
					    __global int *index)
{ 

		int i = get_global_id(0);
		int x = extrema[i*3];
		int y = extrema[i*3+1];
		int z = extrema[i*3+2]+1;
		int p = get_global_size(0);
	    int s = get_global_size(1);
		
		//get the derivative of dD/dx
		
		float D0 = ((float)DoGImg[z*p*s+(x+1)*s+y]-(float)DoGImg[z*p*s+(x-1)*s+y]) / 2.0f;//x
		
		float D1 = ((float)DoGImg[z*p*s+x*s+y+1]-(float)DoGImg[z*p*s+x*s+y-1]) / 2.0f;//y

		float D2 = ((float)DoGImg[(z+1)*p*s+x*s+y]-(float)DoGImg[(z-1)*p*s+x*s+y]) / 2.0f;//sigma
	   

		//get the Hessian Matrix
		int idx =0;
		int h = 1;
        
        float H00 = (float) (DoGImg[z*p*s+(x+h)*s+y]+DoGImg[z*p*s+(x-h)*s+y]-2*DoGImg[z*p*s+x*s+y]); //H00
        float H01 = (float) (DoGImg[z*p*s+(x+h)*s+y+h]+DoGImg[z*p*s+(x-h)*s+y-h]-DoGImg[z*p*s+(x+h)*s+y-h]-DoGImg[z*p*s+(x-h)*s+y+h])/4.0f; //H01
        float H02 = (float) (DoGImg[(z+h)*p*s+(x+h)*s+y]+DoGImg[(z-h)*p*s+(x-h)*s+y]-DoGImg[(z-h)*p*s+(x+h)*s+y]-DoGImg[(z+h)*p*s+(x-h)*s+y])/4.0f; //H02
        float H10 = (float) (DoGImg[z*p*s+(x+h)*s+y+h]+DoGImg[z*p*s+(x-h)*s+y-h]-DoGImg[z*p*s+(x+h)*s+y-h]-DoGImg[z*p*s+(x-h)*s+y+h])/4.0f; //H10
        float H11 = (float) (DoGImg[z*p*s+x*s+y+h]+DoGImg[z*p*s+x*s+y-h]-2*DoGImg[z*p*s+x*s+y]); //H11
        float H12 = (float) (DoGImg[(z+h)*p*s+x*s+y+h]+DoGImg[(z-h)*p*s+x*s+y-h]-DoGImg[(z-h)*p*s+x*s+y+h]-DoGImg[(z+h)*p*s+x*s+y-h])/4.0f; //H12
        float H20 = (float) (DoGImg[(z+h)*p*s+(x+h)*s+y]+DoGImg[(z-h)*p*s+(x-h)*s+y]-DoGImg[(z-h)*p*s+(x+h)*s+y]-DoGImg[(z+h)*p*s+(x-h)*s+y])/4.0f; //H20
        float H21 = (float) (DoGImg[(z+h)*p*s+x*s+y+h]+DoGImg[(z-h)*p*s+x*s+y-h]-DoGImg[(z-h)*p*s+x*s+y+h]-DoGImg[(z+h)*p*s+x*s+y-h])/4.0f; //H21
        float H22 = (float) (DoGImg[(z+h)*p*s+x*s+y]+DoGImg[(z-h)*p*s+x*s+y]-2*DoGImg[z*p*s+x*s+y]);//H22
		
		//inversion of the Hessian

		float det = -H02*H11*H20 + H01*H12*H20 + H02*H10*H21 - H00*H12*H21 - H01*H10*H22 + H00*H11*H22;
		float K00 = H11*H22 - H12*H21;
		float K01 = H02*H21 - H01*H22;
		float K02 = H01*H12 - H02*H11;
		float K10 = H12*H20 - H10*H22;
		float K11 = H00*H22 - H02*H20;
		float K12 = H02*H10 - H00*H12;
		float K20 = H10*H21 - H11*H20;
		float K21 = H01*H20 - H00*H21;
		float K22 = H00*H11 - H01*H10;
		
		//x = -H^(-1)*D
		float solution0 = -(D0*K00 + D1*K01 + D2*K02); //x
		float solution1 = -(D0*K10 + D1*K11 + D2*K12); //y
		float solution2 = -(D0*K20 + D1*K21 + D2*K22); //sigma
		
		//interpolated DoG magnitude at this peak
		
		float dx = DoGImg[z*p*s+x*s+y];
		float Dx = dx + 0.5f * (solution0*D0+solution1*D1+solution2*D2);
		
		float TrH = H00+H11+H22;
		float Tr_D = TrH*TrH;
		
		// use two threshold for selectint the key points
	    if(Dx>0.03&&Tr_D>12.1) //discard the extrema with |D(x)|<0.03 and Tr/D<12.1
		{ 
			idx = atomic_add(index,1);
			//printf("(%d,%d,%d)\n",x,y,z);
			keypoints[idx*3]=x;
			keypoints[idx*3+1]=y;
			keypoints[idx*3+2]=z;
		}
		
}






				

			

