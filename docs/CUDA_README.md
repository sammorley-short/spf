# CUDA README

Below will outline system setups that work for the GPU scripts in this directory (on Mac OS X).

## Common problems

### Xcode Command-Line tools (clang)

The 'Apple clang' provided by the Xcode 9.3 command-line tools doesn't work with CUDA 9.1.128, causing the following error:
```
>>> nvcc fatal   : The version ('90100') of the host compiler ('Apple clang') is not supported
```

This was fixed by installing Xcode 8.3.3 and changing the command-line tools using the command:
```
$ sudo xcode-select -s Applications/Xcode\ 8.3.3.app/Contents/Developer
```

### NVIDIA Drivers on Mac OS X High Sierra 10.13.4 (17E199)

The graphics card drivers shipped with High Sierra do not allow NVIDIA GPU acceleration on older graphics cards (such as NVIDIA GeForce GTX 750).

Instead, a web driver needs to be downloaded and installed. See [here](https://www.tonymacx86.com/threads/nvidia-releases-alternate-graphics-drivers-for-macos-high-sierra-10-13-4-387-10-10-10-30.249039/) for a guide on how to do so.

## Checking versions

The following commands check the versions for the various things mentioned above:

* CUDA (nvcc): `$ nvcc --version`
* Xcode command-line tools: `$ /usr/bin/xcodebuild -version`
* Mac OS X: `$ system_profiler SPSoftwareDataType`
* Python path: `$ python -c "import sys; print('\n'.join(sys.path))"`

## Working setups

### 5/7/18

* *Mac OS X*: macOS 10.13.4 (17E199)
* *NVCC*: V9.1.128
* *CUDA Driver Version*: 387.178
* *GPU Driver Version*: 387.10.10.10.30.106

## Useful links

* [CUDA Drivers for Mac Archive](http://www.nvidia.com/object/mac-driver-archive.html)
* [NVIDIA CUDA Installation Guide for Mac OS X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)
* [Solving NVIDIA Driver Install & Loading Problems](https://www.tonymacx86.com/threads/solving-nvidia-driver-install-loading-problems.161256/)