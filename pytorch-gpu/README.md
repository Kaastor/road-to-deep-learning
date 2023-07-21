# GPU Configuration

## Libraries

- Python 3.8.10 - [Installation guide](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/)
- Update pip: `python3.8 -m pip install --upgrade pip`
- Install tensorflow==2.4.0 & tensorflow-gpu==2.4.0

- [Optional guide](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73?permalink_comment_id=3948350)

## Nvidia Configuration

- [Software requirements](https://www.tensorflow.org/install/gpu#software_requirements)
- Check libraries version on [tensorflow site](https://www.tensorflow.org/install/source#gpu)
- Example for TF=2.4:
  - Python: `3.6-3.8`
  - cuDNN: `8.0`
  - CUDA: `11.0`
  
### Graphics Card Driver

([check CUDA compatibility!](https://docs.nvidia.com/deploy/cuda-compatibility/index.html))
- Install linux-headers: `sudo apt install linux-headers-$(uname -r)`
- Based on above list, [choose correct driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) 
  for your Nvidia Graphics Card
  - Example: Tesla K80, Linux 64-bit, CUDA 11.0
  - Make file executable: `chmod +x driver-file-name.run`
  - Run: `sudo ./driver-file-name.run`
- Install via cmd:
  - ```
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt install nvidia-driver-X nvidia-dkms-X
    ```
- Verify with `nvidia-smi`. Example output:
  ```
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 450.172.01   Driver Version: 450.172.01   CUDA Version: 11.0     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
  | N/A   39C    P0    61W / 149W |      0MiB / 11441MiB |      0%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+
  |   1  Tesla K80           Off  | 00000000:00:05.0 Off |                    0 |
  | N/A   40C    P0    60W / 149W |      0MiB / 11441MiB |     97%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+
  
  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |  No running processes found                                                 |
  +-----------------------------------------------------------------------------+
  ```
  
### CUDA Toolkit

- [Versions list](https://developer.nvidia.com/cuda-toolkit-archive)
- Example: 
 - `sudo wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run`
 - `sudo sh cuda_11.0.2_450.51.05_linux.run`
- During installation **uncheck** GPU Driver
- To uninstall the CUDA Toolkit, run `cuda-uninstaller` in `/usr/local/cuda-11.0/bin`

### cuDNN

- [Versions list for ubuntu1804-x86_64](https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/)
- Make sure that you are downloading correct cuDNN (for proper CUDA version)
- Example - download `libcudnn8` and `libcudnn8-dev` (in that order):
  - `sudo wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb`
  - `sudo dpkg -i libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb`
  - `sudo wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb`
  - `sudo dpkg -i libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb`

- Test with `nvcc -V`

## Verify
Run `gpu-test.py` to check if instance run with GPU or run `docker run --gpus all --rm przomys/gpu-test` on instance 
to check if CUDA is working properly.

## Removing drivers

* `sudo nvidia-uninstall`
* `sudo apt-get remove --purge '^nvidia-.*'`
* `sudo apt-get remove --purge '^libnvidia-.*'`
* `sudo apt-get remove --purge '^cuda-.*'`
