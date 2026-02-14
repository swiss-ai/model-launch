FROM docker.io/lmsysorg/sglang:nightly-dev-20260212-5875ef0a

RUN apt-get update && apt-get install git python3-pip python3-venv -y
RUN pip install git+https://github.com/huggingface/transformers.git@c9ea365a7b56326418769a4ba4682864d407ed63
