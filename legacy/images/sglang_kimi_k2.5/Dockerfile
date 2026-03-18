FROM lmsysorg/sglang:nightly-dev-20260216-d3bae71e

RUN apt-get update && apt-get install git python3-pip python3-venv -y
RUN pip install "sglang @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
RUN pip install nvidia-cudnn-cu12==9.16.0.29
