# Table of contents
# Repository’s structure
# High-level system architecture
# Guide to install and run code
## Installation (for cuda 12.x)
```bash 
conda create -n <your_env> python==3.10
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
python -m pip install vietocr onnx onnxruntime-gpu
```
⚠️ Important Notes

Make sure you are using the correct version of libcudnn for GPU support. You can check your installed version with the following command:
```bash
sudo find / -name "libcudnn.so*" 2>/dev/null
```
Based on your libcudnn version, choose the appropriate versions of onnx and onnxruntime-gpu.

If you want to use the libcudnn bundled with your PyTorch installation, you can set the library path manually like this:
```bash
export LD_LIBRARY_PATH=/home/tuannha/anaconda3/envs/tuannha_onnx/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```
## 🐛 Common Issues
### VietOCR - Notes for ONNX Export

#### ❗ Problem When Exporting to ONNX

During model export using `torch.onnx.export`, the following error may occur when loading the model:

```bash
onnxruntime_pybind11_state.Fail: Load model from ... failed:
Node (/cnn/model/Transpose_1) Op (Transpose) [TypeInferenceError]
Invalid attribute perm {-1, 0, 1}
```

#### 🔍 Reason

PyTorch allows the use of **negative indices** in `permute()` or `transpose()` (e.g., `-1` refers to the last dimension).  
However, **ONNX does not support negative indices** for the `Transpose` operator.

For example, this code is valid in PyTorch:

```python
conv = conv.transpose(-1, -2)
conv = conv.flatten(2)
conv = conv.permute(-1, 0, 1)  # ❌ This causes ONNX export to fail
```
#### ✅ Solution

After install vietocr:
```bash
pip install vietocr
```
Replace all negative indices in permute() or transpose() with explicit positive indices based on the tensor's shape.

📂 **Edit this file:** `<your_env_path>/site-packages/vietocr/model/backbone/vgg.py` or `<your_env_path>/site-packages/vietocr/model/backbone/resnet.py`

```python
conv = conv.transpose(-1, -2)        # still OK in PyTorch and ONNX
conv = conv.flatten(2)
conv = conv.permute(2, 0, 1)         # ✅ ONNX-compatible
```
⚠️ **Important:** If you want to run the model again with PyTorch, make sure to revert the changes and restore the original use of negative indices. These are valid in PyTorch and may be required for correct behavior during standard training or inference.
# Link to a demo video (optional)