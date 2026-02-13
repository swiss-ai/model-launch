FROM docker.io/lmsysorg/sglang:nightly-dev-20260212-5875ef0a

RUN apt-get update && apt-get install git python3-pip python3-venv -y
RUN pip install git+https://github.com/huggingface/transformers.git
# TODO: pin the version @commit_id
