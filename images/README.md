# Swiss AI Model Launch :: Images

This directory is for building custom container images for SLURM nodes. In a glance, these images are built on top of base images (provided by Swiss AI or LLM vendors) with additional dependencies and configurations. The image will be passed on the top of the `.toml` environment configuration file for SLURM job submission. In this way, all nodes in the SLURM cluster will have the same environment and dependencies, ensuring consistency and compatibility.

## Building the Images

The instructions for building custom container images for the SLURM nodes are available in the [documentation](https://github.com/swiss-ai/documentation/blob/main/pages/custom_container.md#create-a-custom-container-image) repository. Change directory (`cd`) to the desired directory and follow the instructions to build the image. A quick summary of the steps to build the image is as follows. **Don't forget to change the paths, image names, and tags as needed.**

```bash
cd /desired/directory/containing/dockerfile                                 # change the path
srun --container-writable --partition normal --account infra01 --pty bash   # instanciate a compute node
podman build -t my_image_name:26.1.0 .                                      # change the name and tag
podman images                                                               # check the built image
enroot import -o $SCRATCH/my_image_name.sqsh podman://my_image_name:26.1.0  # export to enroot format
realpath $SCRATCH/my_image_name.sqsh                                        # get the path of the image
```

## Troubleshooting

1. `Error: no context directory and no Containerfile specified`

   You have missed the trailing dot (`.`) in the `podman build` command. Make sure to include it to specify the current directory as the build context.
