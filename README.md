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

1. Download and setup the Vulkan SDK from https://vulkan.lunarg.com/
  - For Linux distributions, you can either get the essential build requirements from package manager
```shell
dnf install vulkan-headers vulkan-loader-devel
```
```shell
apt-get install libvulkan-dev
```
```shell
pacman -S vulkan-headers vulkan-icd-loader
```

2. Clone this project with all submodules

```shell
git clone https://github.com/nihui/ncnn-vulkan-compute-sample.git
cd ncnn-vulkan-compute-sample
git submodule update --init --recursive
```

3. Build with CMake
  - You can pass -DUSE_STATIC_MOLTENVK=ON option to avoid linking the vulkan loader library on MacOS

```shell
mkdir build
cd build
cmake ..
cmake --build . -j 4
```
