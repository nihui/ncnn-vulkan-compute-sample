# ncnn-vulkan-compute-sample

## Usages

### run fp32 mla test
```shell
mla 0
```

### run fp16 mla test
```shell
mla 1
```

## Build from Source

1. Clone this project with all submodules

```shell
git clone https://github.com/nihui/ncnn-vulkan-compute-sample.git
cd ncnn-vulkan-compute-sample
git submodule update --init --recursive
```

2. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ..
cmake --build . -j 4
```
