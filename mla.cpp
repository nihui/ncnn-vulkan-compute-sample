
#include <benchmark.h>
#include <command.h>
#include <gpu.h>
#include <mat.h>

static const char glsl_data[] = R"(
#version 450

#if NCNN_fp16_storage
#extension GL_EXT_shader_16bit_storage: require
#endif
#if NCNN_fp16_arithmetic
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (constant_id = 0) const int count = 0;
layout (constant_id = 1) const int loop = 1;

layout (binding = 0) readonly buffer a_blob { sfpvec4 a_blob_data[]; };
layout (binding = 1) readonly buffer b_blob { sfpvec4 b_blob_data[]; };
layout (binding = 2) writeonly buffer c_blob { sfpvec4 c_blob_data[]; };

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);

    if (gx >= count || gy >= 1 || gz >= 1)
        return;

    afpvec4 a = buffer_ld4(a_blob_data, gx);
    afpvec4 b = buffer_ld4(b_blob_data, gx);

    afpvec4 c = afpvec4(1.f);

    for (int i = 0; i < loop; i++)
    {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    buffer_st4(c_blob_data, gx, c);
}
)";

int main(int argc, char** argv)
{
    bool use_fp16 = argc > 1 ? atoi(argv[1]) : 0;

    fprintf(stderr, "use_fp16 = %d\n", use_fp16);

    const int count = 10 * 1024 * 1024;
    const int loop = 1000;
    const int cmd_loop = 10;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = use_fp16;
    opt.use_fp16_storage = use_fp16;
    opt.use_fp16_arithmetic = use_fp16;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    // setup pipeline
    ncnn::Pipeline pipeline(vkdev);
    {
        int local_size_x = (int)vkdev->info.subgroup_size();

        pipeline.set_local_size_xyz(local_size_x, 1, 1);

        std::vector<ncnn::vk_specialization_type> specializations(2);
        specializations[0].i = count;
        specializations[1].i = loop;

        // glsl to spirv
        std::vector<uint32_t> spirv;
        ncnn::compile_spirv_module(glsl_data, opt, spirv);

        pipeline.create(spirv.data(), spirv.size() * 4, specializations);
    }

    ncnn::VkAllocator* allocator = vkdev->acquire_blob_allocator();

    // prepare storage
    ncnn::VkMat a;
    ncnn::VkMat b;
    ncnn::VkMat c;
    {
        if (opt.use_fp16_packed || opt.use_fp16_storage)
        {
            a.create(count, 8u, 4, allocator);
            b.create(count, 8u, 4, allocator);
            c.create(count, 8u, 4, allocator);
        }
        else
        {
            a.create(count, 16u, 4, allocator);
            b.create(count, 16u, 4, allocator);
            c.create(count, 16u, 4, allocator);
        }
    }

    for (int i = 0; i < cmd_loop; i++)
    {
        // encode command
        ncnn::VkCompute cmd(vkdev);
        {
            std::vector<ncnn::VkMat> bindings(3);
            bindings[0] = a;
            bindings[1] = b;
            bindings[2] = c;

            std::vector<ncnn::vk_constant_type> constants(0);

            cmd.record_pipeline(&pipeline, bindings, constants, c);
        }

        // time this
        {
            double t0 = ncnn::get_current_time();

            cmd.submit_and_wait();

            double time = ncnn::get_current_time() - t0;

            const double mac = (double)count * (double)loop * 8 * 4 * 2;

            fprintf(stderr, "%f gflops\n", mac / time / 1000000);
        }
    }

    return 0;
}
