#include "driver.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef OSX
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#else
#include <OpenCL/opencl.h>
#endif

void handleCLEnqueueBufferReturn(cl_int err);
void handleCLCreateBufferReturn(cl_int err);
void handleCLSetKernelArg(cl_int err, int index);
void handleClNDRangeKernel(cl_int err);


////////////////////////////////////////////////////////////////////////////////
CLObject* init_driver() {
    CLObject* ocl = (CLObject*)malloc(sizeof(CLObject));
    int err;                            // error code returned from api calls

    unsigned int status[1]={0};               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue command_queue;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input1, input2;                       // device memory used for the input array
    cl_mem output, status_buf;                      // device memory used for the output array

    FILE* programHandle;
    size_t programSize;
    char *programBuffer;
 
    cl_uint nplatforms;
    err = clGetPlatformIDs(0, NULL, &nplatforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error: Failed to get number of platform: %d!\n", err);
        exit(EXIT_FAILURE);

    }

    // Now ask OpenCL for the platform IDs:
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplatforms);
    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error: Failed to get platform IDs: %d!\n",err);
        exit(EXIT_FAILURE);

    }
#ifdef GPU
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
#else    
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
#endif    
    if (err != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Failed to create a device group: %d!\n",err);
        exit(EXIT_FAILURE);

    }

    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        fprintf(stderr,"Error: Failed to create a compute context: %d!\n",err);
        exit(EXIT_FAILURE);

    }

    // Create a command command_queue
    //
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!command_queue)
    {
        fprintf(stderr,"Error: Failed to create a command command_queue: %d!\n",err);
        exit(EXIT_FAILURE);

    }
    // get size of kernel source
    programHandle = fopen("./firmware.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1, (const char**) &programBuffer, &programSize, &err);
    free(programBuffer);
    if (!program)
    {
        fprintf(stderr,"Error: Failed to create compute program: %d!\n",err);
        exit(EXIT_FAILURE);

    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        fprintf(stderr,"Error: Failed to build program executable: %d!\n",err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr,"%s\n", buffer);
        exit(EXIT_FAILURE);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "firmware", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        fprintf(stderr,"Error: Failed to create compute kernel: %d!\n",err);
        exit(EXIT_FAILURE);

    }
    ocl->context = context;
    ocl->command_queue = command_queue;
    ocl->kernel = kernel;
    ocl->program= program;
    ocl->device_id = device_id;

//===============================================================================================================================================================  
// START of assignment code section 

// END of assignment code section 
//===============================================================================================================================================================  
    
    return ocl;
}

int shutdown_driver(CLObject* ocl) {
    int err = clReleaseProgram(ocl->program);
     if (err != CL_SUCCESS) {
            fprintf(stderr,"Error: Failed to release Program: %d!\n",err);
        exit(EXIT_FAILURE);
     }
    err = clReleaseKernel(ocl->kernel);
     if (err != CL_SUCCESS) {
            fprintf(stderr,"Error: Failed to release Kernel: %d!\n",err);
        exit(EXIT_FAILURE);
     }
    err = clReleaseCommandQueue(ocl->command_queue);
     if (err != CL_SUCCESS) {
            fprintf(stderr,"Error: Failed to release Command Queue: %d!\n",err);
        exit(EXIT_FAILURE);
     }
    err = clReleaseContext(ocl->context);
     if (err != CL_SUCCESS) {
            fprintf(stderr,"Error: Failed to release Context: %d!\n",err);
        exit(EXIT_FAILURE);
     }
//===============================================================================================================================================================  
// START of assignment code section      

// END of assignment code section
//===============================================================================================================================================================  
     
    free(ocl);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////

int run_driver(CLObject* ocl,unsigned int buffer_size,  int* input_buffer_1, int* input_buffer_2, int w1, int w2, int* output_buffer) {
    long long unsigned int tid = ocl->thread_num;
#if VERBOSE_MT>2    
     printf("run_driver thread: %llu\n",tid);
#endif
     int err;                            // error code returned from api calls
     int status[1]={-1};               // number of correct results returned
     unsigned int max_iters;
     max_iters = MAX_ITERS;

     size_t global;                      // global domain size for our calculation
     size_t local;                       // local domain size for our calculation

     cl_mem input1, input2;                       // device memory used for the input array
     cl_mem output, status_buf;                      // device memory used for the output array

     // Get the maximum work group size for executing the kernel on the device
     err = clGetKernelWorkGroupInfo(ocl->kernel, ocl->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
     if (err != CL_SUCCESS) {
         fprintf(stderr,"Error: Failed to retrieve kernel work group info! %d\n", err);
         exit(EXIT_FAILURE);
     }

     global = buffer_size; // create as meany threads on the device as there are elements in the array

//===============================================================================================================================================================  
// START of assignment code section 

    // You must make sure the driver is thread-safe by using the appropriate POSIX mutex operations
    // You must also check the return value of every API call and handle any errors 

    // Create the buffer objects to link the input and output arrays in device memory to the buffers in host memory

    cl_int errcode_ret = 0;
    input1 = clCreateBuffer(ocl->context, CL_MEM_USE_HOST_PTR, buffer_size, input_buffer_1, &errcode_ret);
    handleCLCreateBufferReturn(errcode_ret);

    input2 = clCreateBuffer(ocl->context, CL_MEM_USE_HOST_PTR, buffer_size, input_buffer_2, &errcode_ret);
    handleCLCreateBufferReturn(errcode_ret);

    output = clCreateBuffer(ocl->context, CL_MEM_USE_HOST_PTR, buffer_size, output_buffer, &errcode_ret);
    handleCLCreateBufferReturn(errcode_ret);


    int* status_buf_loc = malloc(sizeof(int) * buffer_size);
    status_buf = clCreateBuffer(ocl->context, CL_MEM_USE_HOST_PTR, buffer_size, status_buf_loc, &errcode_ret);
    handleCLCreateBufferReturn(errcode_ret);

    printf("Successfully created buffer objects.\n");


    // Write the data in input arrays into the device memory
    cl_int result;
    result = clEnqueueWriteBuffer(ocl->command_queue, input1, CL_TRUE, 0, buffer_size, &input1, 0, NULL, NULL);
    handleCLEnqueueBufferReturn(result);

    result = clEnqueueWriteBuffer(ocl->command_queue, input2, CL_TRUE, 0, buffer_size, &input2, 0, NULL, NULL);
    handleCLEnqueueBufferReturn(result);

    result = clEnqueueWriteBuffer(ocl->command_queue, output, CL_TRUE, 0, buffer_size, &output, 0, NULL, NULL);
    handleCLEnqueueBufferReturn(result);

    result = clEnqueueWriteBuffer(ocl->command_queue, status_buf, CL_TRUE, 0, buffer_size, &status_buf_loc, 0, NULL, NULL);
    handleCLEnqueueBufferReturn(result);

    printf("Successfully enqueued buffer objects.\n");


    // Set the arguments to our compute kernel
    result = clSetKernelArg(ocl->kernel, 0, sizeof(input1), &input1);
    handleCLSetKernelArg(result, 0);

    result = clSetKernelArg(ocl->kernel, 1, sizeof(input2), &input2);
    handleCLSetKernelArg(result, 1);

    result = clSetKernelArg(ocl->kernel, 2, sizeof(output), &output);
    handleCLSetKernelArg(result, 2);

    result = clSetKernelArg(ocl->kernel, 3, sizeof(status_buf), &status_buf);
    handleCLSetKernelArg(result, 3);

    result = clSetKernelArg(ocl->kernel, 4, sizeof(w1), &w1);
    handleCLSetKernelArg(result, 4);

    result = clSetKernelArg(ocl->kernel, 5, sizeof(w2), &w2);
    handleCLSetKernelArg(result, 5);

    result = clSetKernelArg(ocl->kernel, 6, sizeof(buffer_size), &buffer_size);
    handleCLSetKernelArg(result, 6);


    printf("Successfully set buffer objects in device.\n");

    // Execute the kernel, i.e. tell the device to process the data using the given global and local ranges
    unsigned long global_size = sizeof(global);
    unsigned long local_size = sizeof(local);

    result = clEnqueueNDRangeKernel(ocl->command_queue, ocl->kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    handleClNDRangeKernel(result);

    printf("Successfully loads kernel.\n");

    // Wait for the command commands to get serviced before reading back results. This is the device sending an interrupt to the host    
    

    // Check the status


    // When the status is 0, read back the results from the device to verify the output
   

    // Shutdown and cleanup
    

  
// END of assignment code section 
//===============================================================================================================================================================  
    return *status;

}

void handleClNDRangeKernel(cl_int err){
    if (err != CL_SUCCESS){
        fprintf(stderr,"Error: Failed to execute kernel %d\n", err);
        exit(EXIT_FAILURE);
    }
}


void handleCLSetKernelArg(cl_int err, int index){
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error: Failed to set kernel arg %d! %d\n", index, err);
        exit(EXIT_FAILURE);
    }
}


void handleCLCreateBufferReturn(cl_int err){
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error: Failed to create buffer! %d\n", err);
        exit(EXIT_FAILURE);
    }
}

void handleCLEnqueueBufferReturn(cl_int err){
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error: Failed to enqueue write buffer! %d\n", err);
        exit(EXIT_FAILURE);
    }
}

