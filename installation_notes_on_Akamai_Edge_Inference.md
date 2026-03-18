📦 INSTALLATION.md
Deploying YOLO + CLIP API (api_server.py) on Linode RTX 4000 Ada
🚀 Overview

This guide walks through deploying a GPU-accelerated image search API using:

YOLO (object detection)

CLIP (embedding)

FAISS (vector search)

FastAPI (API server)

Target environment:

Linode GPU instance (RTX 4000 Ada)

Ubuntu 22.04 LTS

0️⃣ Provision Linode GPU Instance

Go to Linode dashboard

Create a GPU instance

Select:

GPU Type: RTX 4000 Ada

OS: Ubuntu 22.04 LTS

Region: closest to your users

Open ports:

22 (SSH)

8000 (API)

1️⃣ Connect via SSH
ssh root@<your-linode-ip>

(Optional but recommended)

adduser yolo
usermod -aG sudo yolo
2️⃣ Install System Dependencies
apt update && apt upgrade -y
apt install -y python3-pip python3-venv git
3️⃣ Install NVIDIA Driver & Verify GPU

Check GPU:

nvidia-smi

If not installed:

apt install -y nvidia-driver-535
reboot

After reboot:

nvidia-smi

Expected: RTX 4000 Ada visible

4️⃣ Set Up Project Directory
mkdir -p /root/yolo-api
cd /root/yolo-api
5️⃣ Create Python Virtual Environment
python3 -m venv venv
source venv/bin/activate
6️⃣ Install Python Dependencies

Upgrade pip:

pip install --upgrade pip
Install PyTorch (CUDA-enabled)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Install Core Dependencies
pip install fastapi uvicorn pillow numpy opencv-python
pip install transformers faiss-cpu
pip install open-clip-torch

Optional (GPU FAISS):

pip install faiss-gpu
7️⃣ Upload Project Files
Option A — SCP
scp -r . root@<ip>:/root/yolo-api/
Option B — Git
git clone <your-repo>
cd <repo>
8️⃣ Project Structure (Expected)
yolo-api/
│
├── api_server.py
├── best.pt
├── embeddings/
├── index.faiss
├── metadata.json
└── venv/
9️⃣ Verify GPU in PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

Expected output:

True
🔟 Run API Server
uvicorn api_server:app --host 0.0.0.0 --port 8000
11️⃣ Test API

From local machine:

curl -X POST \
  -F "image=@plane.jpg" \
  http://<linode-ip>:8000/search
⚙️ Run as Background Service (Recommended)
Option A — tmux
tmux
uvicorn api_server:app --host 0.0.0.0 --port 8000
Option B — systemd

Create service file:

nano /etc/systemd/system/yolo-api.service

Paste:

[Unit]
Description=YOLO API Server
After=network.target

[Service]
User=root
WorkingDirectory=/root/yolo-api
ExecStart=/root/yolo-api/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target

Enable and start:

systemctl daemon-reexec
systemctl daemon-reload
systemctl enable yolo-api
systemctl start yolo-api

Check status:

systemctl status yolo-api
⚡ Performance Optimization
Use GPU explicitly
device = "cuda"
model.to(device)
Disable gradients
with torch.no_grad():
    output = model(img)
Load models at startup
@app.on_event("startup")
def load_models():
    global model
    model = load_model()
Enable FP16 (important for Ada GPUs)
model.half()
Avoid per-request initialization

Bad:

def handler():
    model = load_model()

Good:

model = load_once()
📊 Expected Latency (RTX 4000 Ada)
Stage	Latency
Image decode	5–10 ms
YOLO detection	10–20 ms
CLIP embedding	5–15 ms
FAISS search	<5 ms
Total	30–50 ms
🧠 Key Insight

This system behaves like:

A GPU-resident vector search engine with real-time inference

Performance depends on:

GPU utilization (not CPU)

Batch efficiency

Model warm state

🔧 Future Improvements

Batch multiple requests into one GPU call

Switch to FAISS-GPU

Add request queue (async worker)

Quantize model (INT8 / FP16 tuning)

Deploy behind Nginx reverse proxy

Add authentication layer

Use Triton Inference Server

Multi-GPU scaling

🐛 Troubleshooting
GPU not detected
nvidia-smi

If fails:

reinstall driver

check kernel modules

PyTorch CUDA = False
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Port not reachable
ufw allow 8000
✅ Done

Your API should now be accessible at:

http://<linode-ip>:8000/search