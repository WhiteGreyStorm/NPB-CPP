# Multiproc Project - Harel and Andrew
## IDs 313240806 and 323435420
## Analyzing and Improving the EP kernel
Based on the implementation in https://github.com/GMAP/NPB-CPP

## Code Analysis
We will explain the code for the EP kernel, while going through different spots which can be optimized, either by 
using GPU offload, SIMD or keeping the code CPU parallel (using threads, as the given implementation).

The given implementation starts by calling the math functions so to minimize the effect of paging in later calls
(relevant memory is stored in the shared cache memory).
Then, after initialzing all the variables, the code goes on to the main loop, which is actually the kernel that 
the original writers parallelized using CPU threads.

The number of iterations is decide by splitting the workload into sizes NN and NK which are NN=2^(M-MK) and NK=2^(MK).
NK is the number of random pairs being attempted to be generated (more on that later) and NN is the number of iterations
of the outer loop. The overall workload is of course NN*NK which is 2^(M) = Problem Size.

In the main loop we see the initialization of private variables, among them are statically allocated buffers (that do 
not depend on the problem size). 

The loop logic is pretty simple (which makes it much harder to improve it's efficiency, but we still try) -
We find the seed for the random generation of x's entries 
The qq buffer, which is the private copy of q, stores the number of random pairs actually generated. This in turn is accumulated
in q, after the loop, in a critical section, so that threads that reach this area simultaneously, don't overwrite each others
values. 

We start by initializing the seed for each "batch" of random variables. This batch is all that we care about in each iteration,
thus the initialization is done only once. Afterwards, we actually generate the random variables, that will be used in the 
Marsaglia method (the method this kernel implements). Both of these parts sadly cannot be parallelized, at least not with
great benefit, because of the random generating functions, that have a very serial nature.

## CPU Improvements
However, we managed to notice that the return value of randlc() is actually never used. We tried to remove it
and got a little bit of improvement on the CLASS E problem size. The actual improvemnt was 10 second on average,
with the following run times:

|Iterations 	| With return	| No return|
---             |---            |---       |
| 1              |       343     |		348|
|  2             |       366     |		385|
|   3            |       384     |		346|
|    4           |       367     |		345|
|     5          |       367     |		347|
|      6         |       366     |		366|
|Average		|       365.5	|	356.1666667|

So, this is not the best improvemnt, but it still cuts up some time.
More potential improvements are discussed below.

## Analysis Continues
After the random variables initialization inside of x we go on to the other intersting place in the code -
the actual implementation of the marsaglia method - raising the received random numbers to the power of two,
summing them and checking whether their sum is less than 1, to perform the Marsaglia method computation. 
This part of the code is actually where the reduction from the main loop is needed - this part of the code
is the one modifying the variables sx,sy, on which the reduction accumulation is done.
This is a part where as we see it - a possible optimization can be done.
The code continues to the critical part in which the q array is updated with the values from each thread.
This critical part actually only changes data of a small size (array of size 10), thus we won't optimize it
on the CPU.
Afterwards, we reach the part in which the verification happens and some prints - this part is for sure not
interesting for optimization.

## Another possible CPU optimization
As we mentioned above, the Marsaglia method implementation is a possible spot for optimizations. The calculations
done in this part of the code are mentioned to be non-vectorizeable, however, we believe that this is not the case.
The code in there basically does the following things - 

1. Takes two random variables (two numbers) from the x array -> can be vectorized.
1. Performs a mathematical operation on them -> can be vectorized.
1. Raises each of them to the power of two and sums it (another math operation) -> can be vectorized.
1. Checks if the sum is less than 1 -> a bit problematic, but using masking, as we learned in class - can be vectorized.
1. From here on there are simply other mathematical operations, but this time - they are conditioned on the if statement -> can be vectorized.

As we saw in class, the improvement of the code by using vectorization, is at most improving the performance by 8 for AVX512 (which is our case) for the vectorized code. This can be a huge benefit for us.
Using omp pragmas however, we couldn't succeed in implementing this, so unfortunately, we don't have results for the
overall improvement in performance.

Important note: Since vectorization is very powerful in GPUs, this section here serves us well in the offloading
part of the project.


## GPU Offload
The focus of the work with the offloading is two-fold -
1. Offloading the loops
1. Vectorizing some of the code (as explained in detail above)

Since the second point was explained above, we focus here on the first one.
Offloading the main loop is rather straightforward. We simply need to declare which code will be run on the target and which on the CPU.
The obvious choice is to run the random generation (including the seed initialization) and the vectorizable loop mentioned above - on the GPU, using distribution to teams and multiple GPUs as well. Then, the initialization and the critical region mentioned above, i.e. all the work done on small, fixed-size (non-class dependant) can be done on the CPU.

However, for a very long time we tried to do exactly as written above, but we always ended up with the following error:

`Libomptarget fatal error 1: failure of target construct while offloading is mandatory`

No matter how much we debugged the code, no matter what changes we made, we always ended up running into this error. This of course 
was very much frustrating, because the code, even till this day - doesn't seem to have any problem.
This error is very strnage, since we did manage to do some work on the GPU, even though it's not the most interesting one - the initialization of variables.
Since we wanted to still show how we work with some offloading to the GPU, we analyzed those results in the template report.
Here, we'll analyze what this type of offloading does:
Initializing the variables on the GPU and then not using them, is actually more wastefull than improving, since we waste time on copyinh the data to and from the GPU, even though we don't use it there. However, as we said before, this was only done for proving that we did offload to the GPU.

A little about the synergy between the GPU and the CPU -
As we've seen in class, the GPU can be used with the CPU running in parallel, by combining target pragmas with non-target code, and
utilizing `#pragma omp target data` for defining a block that may have multiple `target` blocks, as well as CPU code. However, in our
case this would probably not be very useful, since the code is very much concentrated in one region - two for loops that can be run 
together on the GPU. Of course, if we want to execute the timer logic done by the original authors - we can do it in the CPU, because 
it requires both the work with files and doing actions that are a very "one time thing" - accessing an array at a single index and writing data to it.