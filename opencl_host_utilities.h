#include <CL/cl.h>

typedef struct opencl_system {
    cl_platform_id platform_id;
    cl_int num_devices;
    cl_device_id *device_ids;
    cl_context context;
    cl_int num_command_queues;
    cl_command_queue *command_queues;
} opencl_system_t;

// Useful OpenCL error code information.
const char *ocl_error_string(cl_int ret);

// Creating and destroying an OpenCL system (platform, device array, context and command queue).
opencl_system_t *create_opencl_system(const cl_device_type device_type, const cl_uint num_devices,
                                      const cl_uint num_command_queues, const char *vendor);
cl_int destroy_opencl_system(opencl_system_t *system);

// Reading a kernel file into a buffer.
char *read_kernel_file(const char *file_name);

