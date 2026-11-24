// y = k*x + 1 on float4 (4 floats per work-item)
__kernel void scale(__global const float4 *in,
                    __global float4 *out, float k)
{
    size_t i = get_global_id(0);
    out[i] = in[i] * k + (float4)(1.0f);
}

// One partial sum per work-group, via local memory
__kernel void sum(__global const float4 *in,
                  __global float *out, __local float *s)
{
    size_t l = get_local_id(0);
    float4 v = in[get_global_id(0)];

    s[l] = v.x + v.y + v.z + v.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t t = get_local_size(0) / 2; t > 0; t >>= 1) {
        if (l < t)
            s[l] += s[l + t];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        out[get_group_id(0)] = s[0];
}
