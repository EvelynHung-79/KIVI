## Setup ENV
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel packaging ninja psutil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.5.6 --no-build-isolation
pip install -e . --no-build-isolation

cd quant && pip install -e . --no-build-isolation
```

## 執行 Unit Test
```bash
python -m unittest discover tests
# or
python -m unittest tests/test_attention_module.py
python -m unittest tests/test_matmul.py
python -m unittest tests/test_quantization.py

```