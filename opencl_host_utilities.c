#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "opencl_host_utilities.h"

/*
 * Return an error string corresponding to the integer error code of a failed OpenCL library call.
 */
const char *ocl_error_string(cl_int ret)
{
    static const char *error_string[] = {"CL_SUCCESS",
                                         "CL_DEVICE_NOT_FOUND",
                                         "CL_DEVICE_NOT_AVAILABLE",
                                         "CL_COMPILER_NOT_AVAILABLE",
                                         "CL_MEM_OBJECT_ALLOCATION_FAILURE",
                                         "CL_OUT_OF_RESOURCES",
                                         "CL_OUT_OF_HOST_MEMORY",
                                         "CL_PROFILING_INFO_NOT_AVAILABLE",
                                         "CL_MEM_COPY_OVERLAP",
                                         "CL_IMAGE_FORMAT_MISMATCH",
                                         "CL_IMAGE_FORMAT_NOT_SUPPORTED",
                                         "CL_BUILD_PROGRAM_FAILURE",
                                         "CL_MAP_FAILURE",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "CL_INVALID_VALUE",
                                         "CL_INVALID_DEVICE_TYPE",
                                         "CL_INVALID_PLATFORM",
                                         "CL_INVALID_DEVICE",
                                         "CL_INVALID_CONTEXT",
                                         "CL_INVALID_QUEUE_PROPERTIES",
                                         "CL_INVALID_COMMAND_QUEUE",
                                         "CL_INVALID_HOST_PTR",
                                         "CL_INVALID_MEM_OBJECT",
                                         "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                                         "CL_INVALID_IMAGE_SIZE",
                                         "CL_INVALID_SAMPLER",
                                         "CL_INVALID_BINARY",
                                         "CL_INVALID_BUILD_OPTIONS",
                                         "CL_INVALID_PROGRAM",
                                         "CL_INVALID_PROGRAM_EXECUTABLE",
                                         "CL_INVALID_KERNEL_NAME",
                                         "CL_INVALID_KERNEL_DEFINITION",
                                         "CL_INVALID_KERNEL",
                                         "CL_INVALID_ARG_INDEX",
                                         "CL_INVALID_ARG_VALUE",
                                         "CL_INVALID_ARG_SIZE",
                                         "CL_INVALID_KERNEL_ARGS",
                                         "CL_INVALID_WORK_DIMENSION",
                                         "CL_INVALID_WORK_GROUP_SIZE",
                                         "CL_INVALID_WORK_ITEM_SIZE",
                                         "CL_INVALID_GLOBAL_OFFSET",
                                         "CL_INVALID_EVENT_WAIT_LIST",
                                         "CL_INVALID_EVENT",
                                         "CL_INVALID_OPERATION",
                                         "CL_INVALID_GL_OBJECT",
                                         "CL_INVALID_BUFFER_SIZE",
                                         "CL_INVALID_MIP_LEVEL",
                                         "CL_INVALID_GLOBAL_WORK_SIZE"};

    return error_string[-ret];
}

/*
 * Based on the name of a vendor name create a basic "OpenCL system" data structure for devices of a chosen type.
 * The function assumes the user wants to have one or more devices from the same vendor and have the same number of
 * queues for each device. The function returns a pointer to a struct containing the platform ID, number of device IDs,
 * an array of device IDs, the context, the number of command queues and an array of command queues. The function
 * helps eliminate boilerplate code in setting up basic contexts and command queues.
 */
opencl_system_t *create_opencl_system(const cl_device_type device_type, const cl_uint num_devices,
                                      const cl_uint num_command_queues, const char *vendor)
{
    opencl_system_t *system;
    cl_int ret;
    cl_uint num_platforms, i, j, num_devices_found;
    cl_platform_id *platform_ids;
    char **platform_vendor_names;
    cl_int found;

    // Check usage. Must an equal number of command queues per device.
    if (num_command_queues % num_devices != 0) {
        fprintf(stderr, "Error creating the requested system: must have an equal number of command queues per device.\n");
        return NULL;
    }

    // Allocate the system data structure we are going to return.
    if ((system = (opencl_system_t *)malloc(sizeof(opencl_system_t))) == NULL) {
        perror("malloc()");
        return NULL;
    }

    // Get the number of platforms.
    num_platforms = 0;
    if ((ret = clGetPlatformIDs(0, NULL, &num_platforms)) != CL_SUCCESS) {
        fprintf(stderr, "Call to clGetPlatformIDs() failed with error code: %s.\n", ocl_error_string(ret));
        free(system);
        return NULL;
    }
    assert(num_platforms != 0);

    // Allocate space for the platforms.
    platform_ids = NULL;
    if ((platform_ids = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
        perror("malloc()");
        return NULL;
    }
    assert(platform_ids != NULL);

    // Get the various platforms.
    if ((ret = clGetPlatformIDs(num_platforms, platform_ids, NULL)) != CL_SUCCESS) {
        fprintf(stderr, "Call to clGetPlatformIDs() failed with error code: %s.\n", ocl_error_string(ret));
        free(system);
        free(platform_ids);
        return NULL;
    }
    assert(platform_ids != NULL);

    // Allocate space for the various platform names.
    platform_vendor_names = NULL;
    if ((platform_vendor_names = (char **)malloc(num_platforms * sizeof(char *))) == NULL) {
        perror("malloc()");
        free(system);
        free(platform_ids);
        free(platform_vendor_names);
        return NULL;
    }
    assert(platform_vendor_names != NULL);
    for (i = 0; i < num_platforms; i++) {
        platform_vendor_names[i] = NULL;
        if ((platform_vendor_names[i] = (char *)malloc(1024 * sizeof(char))) == NULL) {
            perror("malloc()");
            free(system);
            free(platform_ids);
            for (j = 0; j < i; j++)
                free(platform_vendor_names[j]);
            free(platform_vendor_names);
            return NULL;
        }
        assert(platform_vendor_names[i] != NULL);
    }

    // Find the various platform names.
    for (i = 0; i < num_platforms; i++) {
        if ((ret = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, 1024 * sizeof(char),
                                     platform_vendor_names[i], NULL)) != CL_SUCCESS) {
            fprintf(stderr, "Call to clGetPlatformInfo() failed with error code: %s.\n", ocl_error_string(ret));
            free(system);
            free(platform_ids);
            for (i = 0; i < num_platforms; i++)
                free(platform_vendor_names[i]);
            free(platform_vendor_names);
            return NULL;
        }
    }

    // Find the platform associated with the vendor we are interested in.
    found = 0;
    for (i = 0; i < num_platforms; i++) {
        if (strcmp(vendor, platform_vendor_names[i]) == 0) {
            found = 1;
            break;
        }
    }
    if (!found) {
        fprintf(stderr, "Could not find the requested vendor: %s.\n", vendor);
        free(system);
        free(platform_ids);
        for (i = 0; i < num_platforms; i++)
            free(platform_vendor_names[i]);
        free(platform_vendor_names);
        return NULL;
    }
    memcpy(&system->platform_id, &platform_ids[i], sizeof(cl_platform_id));

    // Free all the resources used to find the platform now that we have the one we want.
    free(platform_ids);
    for (i = 0; i < num_platforms; i++)
        free(platform_vendor_names[i]);
    free(platform_vendor_names);

    // Mark the number of devices.
    system->num_devices = num_devices;

    // Allocate space for the device ID array.
    system->device_ids = NULL;
    if ((system->device_ids = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id))) == NULL) {
        perror("malloc");
        free(system);
        return NULL;
    }
    assert(system->device_ids != NULL);

    // Get the device IDs.
    if ((ret = clGetDeviceIDs(system->platform_id, device_type, num_devices, system->device_ids, &num_devices_found))
        != CL_SUCCESS) {
        fprintf(stderr, "Call to clGetDeviceIDs() failed with error code: %s.\n", ocl_error_string(ret));
        free(system->device_ids);
        free(system);
        return NULL;
    } else if (num_devices_found != num_devices) {
        fprintf(stderr, "Call to clGetDeviceIDs() could not get the desired number of devices (%u). Only found %u.\n",
                num_devices, num_devices_found);
        free(system->device_ids);
        free(system);
        return NULL;
    }

    // Create the context.
    // TODO: Consider adding callback functionality for error handling.
    if ((system->context = clCreateContext(NULL, num_devices, system->device_ids, NULL, NULL, &ret)) == NULL) {
        fprintf(stderr, "Call to clCreateContext() failed with error code: %s.\n", ocl_error_string(ret));
        free(system->device_ids);
        free(system);
        return NULL;
    }

    // Mark the number of command queues.
    system->num_command_queues = num_command_queues;

    // Allocate space for the command queue array.
    system->command_queues = NULL;
    if ((system->command_queues = (cl_command_queue *)malloc(num_command_queues * sizeof(cl_command_queue))) == NULL) {
        perror("malloc");
        free(system->device_ids);
        if ((ret = clReleaseContext(system->context)) != CL_SUCCESS)
            fprintf(stderr, "Call to clReleaseContext() failed with error_code: %s.\n", ocl_error_string(ret));
        free(system);
        return NULL;
    }
    assert(system->command_queues != NULL);

    // Create the command queues.
    // TODO: Consider adding command queue properties functionality.
    for (i = 0; i < num_command_queues; i++) {
        if ((system->command_queues[i] = clCreateCommandQueue(system->context, system->device_ids[i / num_devices],
                                                              0, &ret)) == NULL) {
            fprintf(stderr, "Call to clCreateCommandQueue() failed with error code: %s.\n", ocl_error_string(ret));
            free(system->device_ids);
            for (j = 0; j < i; j++) {
                if ((ret = clReleaseCommandQueue(system->command_queues[j])) != CL_SUCCESS)
                    fprintf(stderr, "Call to clReleaseCommandQueue() failed with error code: %s.\n", ocl_error_string(ret));
            }
            if ((ret = clReleaseContext(system->context)) != CL_SUCCESS)
               fprintf(stderr, "Call to clReleaseContext() failed with error_code: %s.\n", ocl_error_string(ret));
            free(system->command_queues);
            free(system);
            return NULL;
        }
        assert(system->command_queues[i] != NULL);
    }

    // Return the initialized OpenCL system data structure.
    return system;
}

/*
 * Free the host and device resources associated with an "OpenCL system".
 */
cl_int destroy_opencl_system(opencl_system_t *system)
{
    cl_int i, ret;

    // Release the command queues.
    for (i = 0; i < system->num_command_queues; i++) {
        if ((ret = clReleaseCommandQueue(system->command_queues[i])) != CL_SUCCESS) {
            fprintf(stderr, "Call to clReleaseCommandQueue() failed with error code: %s.\n", ocl_error_string(ret));
            return -1;
        }
    }

    // Release the context.
    if ((ret = clReleaseContext(system->context)) != CL_SUCCESS) {
        fprintf(stderr, "Call to clReleaseContext() failed with error_code: %s.\n", ocl_error_string(ret));
        return -1;
    }

    // Free the system resources.
    free(system->device_ids);
    free(system);

    // Return successully.
    return 0;
}

/*
 * Return a pointer to a buffer containing the kernel file whose name is passed as an argument.
 */
char *read_kernel_file(const char *file_name)
{
    FILE *kernel_file;
    long kernel_size;
    char *kernel_source;

    // Open the kernel file.
    kernel_file = NULL;
    if ((kernel_file = fopen(file_name, "r")) == NULL) {
        perror("fopen()");
        return NULL;
    }
    assert(kernel_file != NULL);

    // Find the end of the file.
    if (fseek(kernel_file, 0, SEEK_END) < 0) {
        perror("fseek()");
        return NULL;
    }

    // Report the size of the file.
    if ((kernel_size = ftell(kernel_file)) < 0) {
        perror("ftell()");
        return NULL;
    }

    // Return the seek position to the beginning of the file.
    if (fseek(kernel_file, 0, SEEK_SET) < 0) {
        perror("fseek()");
        return NULL;
    }

    // Allocate space for the kernel source.
    kernel_source = NULL;
    if ((kernel_source = (char *)malloc(kernel_size + 1)) == NULL) {
        perror("malloc()");
        return NULL;
    }
    assert(kernel_source != NULL);

    // Read the kernel source file into the newly allocated buffer.
    if (fread(kernel_source, sizeof(char), kernel_size, kernel_file) < (size_t)kernel_size) {
        perror("fread()");
        free(kernel_source);
        return NULL;
    }

    // Close the kernel source file.
    if (fclose(kernel_file) < 0) {
        perror("fclose()");
        free(kernel_source);
        return NULL;
    }

    // Null terminate the kernel source string.
    kernel_source[kernel_size] = '\0';

    // Return the kernel source string.
    return kernel_source;
}

