# Nvidia-on-Ubuntu-20.04
![image](https://github.com/FilmBuachoom/Nvidia-on-Ubuntu-20.04/assets/109780340/28fd6548-0721-41b2-83df-63d26e47a9a5)

Installing NVIDIA drivers is crucial for TensorFlow GPU support, enabling GPU acceleration, providing CUDA and cuDNN libraries, ensuring compatibility and stability, and utilizing extra GPU features.

## Literature Review (Compiled by ChatGPT)
* **NVIDIA driver** <br>
The NVIDIA driver is software developed by NVIDIA Corporation specifically for NVIDIA graphics processing units (GPUs). It acts as a communication bridge between the operating system and the GPU hardware, providing an interface for software applications to utilize the GPU's capabilities. The driver includes libraries, APIs, and low-level components that enable GPU acceleration and enable features such as parallel processing, high-performance computing, and graphics rendering.
* **CUDA** <br>
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to harness the power of NVIDIA GPUs for general-purpose computing tasks beyond graphics processing. CUDA provides a programming interface and a set of libraries that enable developers to write GPU-accelerated code. It includes a parallel computing architecture, a C/C++ compiler, and a runtime system that facilitates the execution of code on the GPU. TensorFlow, as well as other deep learning frameworks, can use CUDA to accelerate computations on NVIDIA GPUs.
* **cuDNN** <br>
cuDNN (CUDA Deep Neural Network) is a GPU-accelerated library developed by NVIDIA specifically for deep neural network operations. It provides highly optimized implementations of mathematical functions and algorithms commonly used in deep learning, such as convolution, pooling, normalization, and activation functions. cuDNN is designed to work in conjunction with CUDA and NVIDIA GPUs, offering significant performance improvements for deep learning frameworks like TensorFlow. It helps to accelerate the training and inference processes of deep neural networks by leveraging the parallel processing capabilities of NVIDIA GPUs.
* **NVIDIA Docker runtime** <br>
NVIDIA Docker runtime is a software component that integrates Docker containers with NVIDIA GPUs. It enables GPU-accelerated applications to run in containers, providing access to the GPU hardware and libraries, and facilitating the deployment of GPU-accelerated workloads in a containerized environment.

## Installing
1. Choosing Nvidia and TensorFlow dependencies
2. Install Nvidia Driver
3. Install CUDA Toolkit
4. Install cuDNN
5. Add the CUDA environment variables
6. Verify CUDA installation
7. Install the NVIDIA Docker runtime (optional)

## Uninstalling
