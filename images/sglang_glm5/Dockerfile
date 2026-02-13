FROM docker.io/lmsysorg/sglang:nightly-dev-20260213-a0ebaa64

# setup
RUN apt-get update && apt-get install git python3-pip python3-venv -y
        
# Update flash-attn.
RUN pip install git+https://github.com/huggingface/transformers.git
# TODO: pin the version @commit_id
