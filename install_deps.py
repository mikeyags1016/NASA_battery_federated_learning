import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.10.0",
    "torchvision==0.25.0",
    "torchaudio==2.10.0",
    "--index-url", "https://download.pytorch.org/whl/cpu"
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "flwr==1.14.0",
    "flwr-datasets==0.5.0",
    "grpcio>=1.60.0,<1.65.0",
    "grpcio-status>=1.60.0,<1.65.0",
    "grpcio-health-checking>=1.60.0,<1.65.0",
    "protobuf>=4.21.0,<6.0.0",
    "tomli>=2.0.0",
    "ray==2.31.0",
    "--force-reinstall", "--no-deps"
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "pycryptodome",
    "iterators"
], check=True)

print("✅ Environment ready.")