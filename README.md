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
    * Check GPU <br>
        My gpu is `Nvidia Tesla T4`.

    * Make sure your GPU supports CUDA version ([checked here](https://www.nvidia.com/download/index.aspx)) <br>
        ![5c5399db-f732-4028-92e4-ceb0e7dce300](https://github.com/FilmBuachoom/Nvidia-on-Ubuntu-20.04/assets/109780340/3017252a-36de-49e8-bc03-b07e3b94ca46)

        In this case it is used as `CUDA Toolkit 12.0`. CUDA Toolkit 12.0 will be bundled with Nvidia-driver 525. CUDA Toolkit 12.0 is chosen because it uses NVIDIA Docker runtime which will support CUDA version 11.8 and above. Of course, TensorFlow is not supported with CUDA Toolkit 12.0 `in the process of installing CUDA Toolkit will use version 11.2 instead.`

    * Select the version of TensorFlow that supports CUDA version ([checked here](https://www.tensorflow.org/install/source#gpu)) <br>
        As it uses `CUDA Toolkit 11.2`, TensorFlow can be used from `2.5.0 - 2.11.0`. <br>
        For `TensorFlow 2.5.0 - 2.11.0`, `CUDA Toolkit 11.2` and `cuDNN 8.1` are supported.

    * Verify that Nvidia-driver supports CUDA Toolkit version ([checked here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#default-to-minor-version)) <br>
        ![image]()

    * Summary software dependencies <br>
        * Nvidia-driver 525
        * tensorflow==2.5.0 - tensorflow==2.11.0
        * CUDA Toolkit 11.2
        * cuDNN 8.1

2. Install Nvidia Driver 525
    * Add the Graphics Drivers PPA (Personal Package Archive) to your system
        ```
        sudo add-apt-repository ppa:graphics-drivers/ppa
        sudo apt-get update
        ```
    * Install the driver 525.105.17 package
        ```
        sudo apt-get install nvidia-driver-525
        ```
    * Reboot your system to load the newly installed NVIDIA driver
        ```
        sudo reboot
        ```
    * Verify installation
        ```
        nvidia-smi
        ```
        ![image]()
3. Install CUDA Toolkit
    * Download CUDA 11.2 Toolkit ([CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive))
        ```
        sudo wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
        ```
    * Install the NVIDIA CUDA 11.2 Toolkit (DO NOT check the option of installing the driver!!!)
        ```
        sudo sh cuda_11.2.2_460.32.03_linux.run
        ```
        ![image]()
4. Install cuDNN 8.1.1
    * Install prerequisites library
        ```
        sudo apt-get install zlib1g
        ```
    * Download CuDNN 8.1.1 (for CUDA 11.0,11.1 and 11.2) ([NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive))
        ```
        sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-linux-x64-v8.1.1.33.tgz
        ```
    * Unzip the cuDNN package
        ```
        sudo tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
        ```
    * Copy the following files into the CUDA toolkit directory
        ```
        sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include
        sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
        sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
        ```
5. Add the CUDA environment variables <br>
    Create a shell script to declare the address of the cuda folder. `activate-cuda.sh` and save to `/etc/profile.d/`
    ```
    #!/bin/bash

    # Add the CUDA environment variables + set .sh file @ /etc/profile.d/
    echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```
6. Verify CUDA installation
    * Verify CUDA installation
        ```
        nvcc -V
        ```
        ![image]()

    * Test Tensorflow version 2.5 - 2.11
        ```
        pip3 install -q tensorflow==2.11.0
        python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
        ```
7. Install the NVIDIA Docker runtime (optional) <br>
    NVIDIA Docker runtime for servers that use docker (when there are services that require a graphics card).
    * Add the NVIDIA Docker repository
        ```
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        ```
    * Install the NVIDIA Docker runtime
        ```
        sudo apt-get install -y nvidia-docker2
        ```
    * Restart Docker service
        ```
        sudo systemctl restart docker
        ```
    * Verify installation
        ```
        docker run --gpus all nvidia/cuda:11.2.0-base nvidia-smi
        ```
        ![image]()

## Uninstalling
1. Uninstall Nvidia driver
    ```
    sudo apt-get --purge remove -y "*nvidia*" "libxnvctrl*"
    sudo apt-get remove --purge '^nvidia-.*'
    ```
2. To remove CUDA Toolkit
    ```
    sudo apt-get --purge remove -y "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
    "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    sudo apt-get remove --purge '^cuda-.*'
    ```
3. Uninstall Nvidia CUDA
    ```
    cd /usr/local/
    sudo rm -rf cuda-11_2
    sudo rm -rf cuda
    ```
4. Remove any references to CUDA from the environment variables <br>
    > Delete PATH=/usr/local/cuda-11.2/bin:$PATH` from ~/.bashrc

    > Delete LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH` from ~/.bashrc
    ```
    sudo nano ~/.bashrc
    ```
5. To clean up the uninstall
    ```
    sudo apt-get -y autoremove
    ```
6. Re-checked <br>
    After running both commands, no results are displayed. It will be considered that the deletion is complete.
    ```
    dpkg -l | grep cuda
    dpkg -l | grep nvidia
    ```
7. Reboot
    ```
    sudo reboot
    ```
