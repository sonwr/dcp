# Deep Closest Point (DCP)

This repository contains the implementation of the paper **"Deep Closest Point: Learning Representations for Point Cloud Registration"** by Yue Wang et al. The original code is available at [Yue Wang's GitHub](https://github.com/WangYueFt/dcp).

## Modifications for Windows Environment

This fork adapts the original Linux-based code to run on a Windows environment. Below are the details of the modifications and the environment in which the code was tested.

### Experimental Environment

- **Processor:** 13th Gen Intel(R) Core(TM) i7-13700F 2.10 GHz
- **RAM:** 32.0 GB
- **OS:** Windows 11 Pro (64-bit operating system, x64-based processor)
- **GPU:** NVIDIA GeForce RTX 4060 Ti
- **CUDA:** 11.8

### Python Packages

- **Python:** 3.9.13
- **Dependencies:**
  ```bash
  pip install torch scipy numpy h5py tqdm tensorboardX
  ```

### Modifications

1. **OS-Dependent Data Handling:**
   - Adjusted code to handle Windows-specific file paths and operations.

2. **Deprecated Functions:**
   - Replaced deprecated SciPy functions with their updated equivalents.
   - Example:
     ```python
     r = Rotation.from_dcm(mats[i])
     ```
     updated to:
     ```python
     r = Rotation.from_matrix(mats[i])
     ```

3. **Optimizer and Scheduler Call Order:**
   - Corrected the order of `scheduler.step()` and `optimizer.step()` to ensure the learning rate scheduler works as intended.
   - Updated order:
     ```python
     optimizer.step()
     scheduler.step()
     ```

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sonwr/dcp.git
   cd dcp
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.9.13 installed. Then, install the required packages:
   ```bash
   pip install torch scipy numpy h5py tqdm tensorboardX
   ```

3. **Prepare Data:**
   - Ensure your data is correctly formatted and accessible. Modify any paths in the code as necessary to point to your data directories.

4. **Run Training:**
   - DCP-v1
   ```bash
   python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval
   ```
   
   - DCP-v2
   ```bash
   python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval
   ```

5. **Run Testing:**
   - DCP-v1
   ```bash
   python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy
   ```

   - DCP-v2
   ```bash
   python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy
   ```

## Notes

- Ensure your CUDA version matches the version of PyTorch you are using.
- Modify any additional file paths or environment-specific settings as necessary.

## Acknowledgements

The original implementation of this project was done by Yue Wang et al. and is available [here](https://github.com/WangYueFt/dcp). This fork adapts the code to be compatible with Windows OS and includes other minor improvements.

For more details on the paper, refer to the original publication: [Deep Closest Point: Learning Representations for Point Cloud Registration](https://arxiv.org/abs/1905.03304).

## License
MIT License
