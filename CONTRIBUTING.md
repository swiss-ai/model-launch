# Contributing Guidelines

## Understanding the Flow of Running a Model

Once you run `sml` to execute a model, the following steps occur:

1. A job is submitted to the target cluster and will be queued.
2. Once the job get the allocated resources, the requested nodes will be allocated. The nodes will be initialized on based on the provided `toml` file. The `toml` configuration file is used to specify the base (Docker) image, storage bindings, and other necessary environment variables for the model execution.
3. The model will be executed either with `vllm` or `sglang`.

## Using a New Model

Based on the [flow](#understanding-the-flow-of-running-a-model) described above, to use a new model, you need to:

1. Create the docker image.
2. Create the `toml` configuration file for the model.
3. Fabricate the `sml` command to run the model.

As we are providing a set of base images and associated `toml` files, for most of the models, you only need to fabricate the `sml` command to run the model. However, for some of the usecases, you may need to create a new docker image and the corresponding `toml` file.

### Fabricating the `sml` Command

The `sml` advanced mode gives you the full control over the execution of the model in the CLI. The usage of `sml` advanced command is pretty similar to using `vllm` or `sglang` command themselves and it is as easy as follows:

1. Find the `vllm` or `sglang` command to run the model you want to run. You can usually find them in the HuggingFace page or the official documentation of the model.
2. Specify the `sml`-specific arguments such as `--slurm-nodes`, `--serving-framework`, `--slurm-environment`, and the others which are specified in [README](README.md#advanced-usage).
3. Pass the arguments of the base `vllm` or `sglang` command to the `--framework-args` argument of the `sml` command. Make sure that you are setting the host to `0.0.0.0` and the port to `8080` in the `--framework-args`.

    ```bash
    sml advanced \
    --slurm-nodes 1 \
    --serving-framework sglang \
    --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
    --framework-args "--model-path swiss-ai/Apertus-8B-Instruct-2509 \
        --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
        --host 0.0.0.0 \
        --port 8080"
    ```
