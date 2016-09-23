A Visual Tutorial on 2D Cross Correlation in C++ and Python
-------------------------------------------------------------

I have a program that uses OpenCV to compute either the convolution or cross-correlation of an image with a specified kernel. I compute cross-correlation by setting the conjB flag to true when calling [cv::mulSpectrums][1].

Initially, convolution consistently produced correct results, but the result of cross-correlation was always wrong.

The only difference is that the kernel spectrum is conjugated before multiplying. Both were computed using `cv::dft`, and so are in the `CCS-packed` format expected by `mulSpectrums`. 

I compared the result with the output of my Python code using `scipy.signal.correlate`.

```C++
cv::dft( dft_buf, dft_buf, 1, img_in.rows );
cv::mulSpectrums( dft_buf, kernel_dft, dft_buf, 0, CORRELATE );
dft( dft_buf, dft_buf, cv::DFT_INVERSE + cv::DFT_SCALE, kernel.rows/2 + img_in.rows );

int col_offset = ( kernel.cols % 2 ) ? ( kernel.cols/2 ) : ( kernel.cols/2 - 1 ); // is odd?
int row_offset = ( kernel.rows % 2 ) ? ( kernel.rows/2 ) : ( kernel.rows/2 - 1 );

cv::Mat result_roi( dft_buf, cv::Rect( col_offset, row_offset, img_in.cols, img_in.rows ) );
result_roi.copyTo( img_out );
```

  [1]: http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#mulspectrums

I tried flipping the the kernel before computing the DFT, and that produced the correct cross-correlation. 

On closer inspection, I discovered that the erroneous correlation result resembles the correct result, but shifted up and to the left. The former was displayed in scientific notation, so it was hard to see the pattern at first.

The reason is that taking the conjugate of a zero-padded kernel flips the whole zero-padded kernel, and not just the original kernel (which is what we want). That makes sense, because correlation is the reverse of convolution, i.e. conjugating the DFT cancels out the flipping of convolution. Basically, you're pre-flipping the kernel and then convolving it with the image. 

Because the result is a shifted version of the correct result, the naive solution would be to simply select another ROI from the output. However, the result is actually shifted circularly, so that wouldn't work. The correct result is present, but disjoint. 

There are two possible solutions: 

1. Flip the kernel before zero-padding it and then compute the DFT, 
2. Change the location of the image in the zero-pad buffer.

I chose to do the latter. The idea is that, by shifting the image you also shift the result, so that all elemements in the result are contained in one region of interest.

2D circular convolution can be visualized as a plain containing infinite adjacent copies of the zero-padded image. Rather than wrapping the zero-padded kernel around, it can be visualized as linear convolution with the lower-right copy, with adjacent copies providing the "history."

     111100000 | 111100000 
     111100000 | 111100000 
     000000000 | 000000000 
     000000000 | 000000000 
    -----------|-----------
     111100000 | 111100000 
     111100000 | 111100000 
     000000000 | 000000000 
     000000000 | 000000000 

For correlation with a 2x2 kernel, you would change it to this:

    000000000 | 000000000
    011110000 | 011110000
    011110000 | 011110000
    000000000 | 000000000
    ----------|-----------
    000000000 | 000000000
    011110000 | 011110000
    011110000 | 011110000
    000000000 | 000000000

You might think that the image should actually be shifted down and to the right by one more space, so that the "wrapped" kernel buffer will start with the kernel overlapping only the upper-left element of the image. The idea being that, convolution will first flip the conjugate kernel to look like this:

    110000000
    110000000
    000000000
    000000000

However, this is incorrect. The conjugate symmetry of the Fourier Transform states that flipping a signal, or an image, is equivalent to taking the complex conjugate of its Fourier Transform. However, the computer representation of the time-series is different than expected. For a 1D signal:

    [ 1, 2, 3, 4, 0, 0 ]

flips to:

    [ 1, 0, 0, 4, 3, 2 ]

and not:

    [ 0, 0, 4, 3, 2, 1 ]
    
The reason is that the value of x[n] at n=0 must remain the same after flipping, i.e. it flips about the origin. The values to the right of n=0 can be thought of as being for n < 0. So the array of corresponding indices would be:

    [ 0, -5, -4, -3, -2, -1 ]
    
Again, the key is to think of a circular axis. 

Carrying this over to 2D, after zero-padding our 2x2 kernel and conjugating its DFT, this is what it would look like in the spatial domain:

    100000001
    000000000
    000000000
    100000001

It's difficult to visualize how to index the correlation using this kernel. However, in 1D, this can be interpreted as flipping the array, and then circular shifting by one to the right. In 2D, it flips the array vertically and horizontally, and then circular shifts it down and right by one.

 Recall that the image needs to be shifted in the buffer for the result to be contained in a single ROI. To acheive this, the upper left element of the image is placed at `(kernel.cols-1, kernel.rows-1)`, instead of at `(0. 0)`. The upper-left element of the kernel is placed at `(0, 0)`, but taking the conjugate flips it vertically and horizontally about `(0, 0)`, as previously explained. Computing the DFTs of both and then multiplying them is equivalent to circular convolution in the spatial domain, which means that the kernel is implicitly flipped again (in the spatial domain).

Note that the upper left element of the result will be at `(0, 0)`, and not at `(kernel.cols-1, kernel.rows-1)`. This may seem counter-intutive, because the leading edge of the flipped kernel will have to shift to `(kernel.cols-1, kernel.rows-1)` before it begins to eclipse the image. However, the trailing portion of the kernel - i.e. the part that wraps around diagonally - eclipses the image at the beginning of the convolution. So it will be the trailing portion of the kernel, and not the leading edge, that "slides" across the image.

Consider this example image, already copied into the zero-pad buffer:

    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 1 1 1 1 1 1
    0 0 0 1 1 1 1 1 1
    0 0 0 1 1 1 1 1 1

And we'll correlate it with this kernel:

    11 12 13 14
    21 22 23 24
    31 32 33 34
    41 42 43 44
    
After zero-padding the kernel and conjugating it's DFT, the spatial representation is:
    
    22 23 24 0 0 0 0 0 21
    32 33 34 0 0 0 0 0 31
    42 43 44 0 0 0 0 0 41
     0  0  0 0 0 0 0 0  0
     0  0  0 0 0 0 0 0  0
    12 13 14 0 0 0 0 0 11

This is hard to really show without an animation, but this is the best I could do here.

Here's the code:
```C++
// copy image to buffer
int col_offset = ( CORRELATE ) ? ( kernel.cols - 1 ) : ( 0 );
int row_offset = ( CORRELATE ) ? ( kernel.rows - 1 ) : ( 0 );
cv::Mat image_roi( dft_buf, cv::Rect( col_offset, row_offset, img_in.cols, img_in.rows ) );
img_in.copyTo( image_roi );

// Compute DFT of image, and multiply it with kernel DFT
int non_zero_rows = ( CORRELATE ) ? ( img_in.rows + kernel.rows - 1 ) : ( img_in.rows );
cv::dft( dft_buf, dft_buf, 0, non_zero_rows );
cv::mulSpectrums( dft_buf, kernel_dft, dft_buf, 0, CORRELATE );

// Compute IDFT
dft( dft_buf, dft_buf, cv::DFT_INVERSE + cv::DFT_SCALE, kernel.rows/2 + img_in.rows );

// Select ROI from result, and either copy or pass reference to output buffer
col_offset = ( kernel.cols % 2 ) ? ( kernel.cols/2 ) : ( kernel.cols/2 - 1 ); // is odd?
row_offset = ( kernel.rows % 2 ) ? ( kernel.rows/2 ) : ( kernel.rows/2 - 1 );
cv::Mat result_roi( dft_buf, cv::Rect( col_offset, row_offset, img_in.cols, img_in.rows ) );
result_roi.copyTo( img_out );
```
