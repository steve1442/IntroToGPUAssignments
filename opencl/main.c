
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define K 2.0f
#define MAXP 16

static cl_context ctx;
static cl_command_queue q;

static cl_device_id pick(void)
{
    cl_platform_id p[4];
    cl_uint n = 0, i;
    cl_device_id d;
    char name[128];

    clGetPlatformIDs(4, p, &n);
    for (i = 0; i < n && i < 4; i++) {
        if (clGetDeviceIDs(p[i], CL_DEVICE_TYPE_GPU, 1, &d, NULL) == 0) {
            clGetDeviceInfo(d, CL_DEVICE_NAME, 128, name, NULL);
            printf("Device: %s\n", name);
            return d;
        }
    }
    fprintf(stderr, "no GPU found\n");
    exit(1);
}

static cl_program build(cl_device_id d)
{
    cl_int e;
    char log[2048] = "", src[4096] = "";
    const char *s = src;
    FILE *f = fopen("kernel.cl", "r");
    cl_program p;

    if (!f || !fread(src, 1, sizeof src - 1, f))
        exit(puts("cannot read kernel.cl"));
    fclose(f);
    p = clCreateProgramWithSource(ctx, 1, &s, NULL, &e);
    e;
    e = clBuildProgram(p, 1, &d, "", NULL, NULL);
    clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, sizeof log, log, 0);
    if (e != CL_SUCCESS)
        fprintf(stderr, "%s\n", log);
    e;
    return p;
}

static cl_mem sub(cl_mem b, size_t off, size_t sz)
{
    cl_buffer_region r = { off, sz };
    cl_int e;
    cl_mem m = clCreateSubBuffer(b, 0, CL_BUFFER_CREATE_TYPE_REGION,
                                 &r, &e);

    e;
    return m;
}

// One launch per part. each part sees only its sub-buffer
static void scale(cl_kernel k, cl_mem in, cl_mem out, size_t n,
                  size_t ls, int np, cl_event *ev)
{
    float kk = K;
    size_t pb = n * sizeof(float) / np, gs = n / 4 / np;

    for (int i = 0; i < np; i++) {
        cl_mem a = sub(in, i * pb, pb), b = sub(out, i * pb, pb);

        clSetKernelArg(k, 0, sizeof a, &a);
        clSetKernelArg(k, 1, sizeof b, &b);
        clSetKernelArg(k, 2, sizeof kk, &kk);
        clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL,
                                  &ev[i]);
        clReleaseMemObject(a);
        clReleaseMemObject(b);
    }
}

// Waits on the scale events, then sums the per-group partials
static double reduce(cl_kernel k, cl_mem out, size_t n, size_t ls,
                     int np, cl_event *ev)
{
    size_t gs = n / 4, ng = gs / ls;
    cl_int e;
    double t = 0;
    cl_mem pm = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, ng * 4, 0, &e);
    float *p;

    e;
    clSetKernelArg(k, 0, sizeof out, &out);
    clSetKernelArg(k, 1, sizeof pm, &pm);
    clSetKernelArg(k, 2, ls * sizeof(float), NULL);
    clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, np, ev, NULL);
    p = clEnqueueMapBuffer(q, pm, CL_TRUE, CL_MAP_READ, 0, ng * 4,
                           0, NULL, NULL, &e);
    e;
    for (size_t i = 0; i < ng; i++)
        t += p[i];
    clEnqueueUnmapMemObject(q, pm, p, 0, NULL, NULL);
    clFinish(q);
    clReleaseMemObject(pm);
    return t;
}

static double ms(cl_event *ev, int np)
{
    cl_ulong a, b;
    double t = 0;

    for (int i = 0; i < np; i++) {
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, 8,
                                &a, NULL);
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, 8,
                                &b, NULL);
        t += (b - a) * 1e-6;
    }
    return t;
}

static int go(size_t n, size_t ls, int np)
{
    cl_device_id d = pick();
    cl_int e;
    cl_event ev[MAXP];
    float *x = malloc(n * sizeof *x);
    double want = 0, got;
    cl_program p;
    cl_kernel ks, kr;
    cl_mem in, out;

    for (size_t i = 0; i < n; i++) {
        x[i] = (float)(i % 4);
        want += K * x[i] + 1.0;
    }
    ctx = clCreateContext(NULL, 1, &d, NULL, NULL, &e);
    e;
    q = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &e);
    e;
    p = build(d);
    ks = clCreateKernel(p, "scale", &e);
    e;
    kr = clCreateKernel(p, "sum", &e);
    e;
    in = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, n * 4, x, &e);
    e;
    out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * 4, NULL, &e);
    e;
    scale(ks, in, out, n, ls, np, ev);
    got = reduce(kr, out, n, ls, np, ev);
    printf("sum=%.0f want=%.0f scale=%.3f ms\n", got, want, ms(ev, np));
    free(x);
    return fabs(got - want) <= 1e-6 * want;
}

int main(int argc, char **argv)
{
    size_t n = argc > 1 ? strtoul(argv[1], 0, 10) : 1048576;
    size_t ls = argc > 2 ? strtoul(argv[2], 0, 10) : 128;
    int np = argc > 3 ? atoi(argv[3]) : 4;

    if (np < 1 || np > MAXP || !ls || (ls & (ls - 1)) ||
        n % (4 * np * ls)) {
        fprintf(stderr, "need: n %% (4*parts*local)==0, local 2^k\n");
        return 1;
    }
    return go(n, ls, np) ? 0 : 1;
}
