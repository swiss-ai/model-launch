# How to use Harbor with OpenCode

Follow these steps:

1. Check if the desired model is already running on the [API](https://serving.swissai.cscs.ch/).
2. If it is not, launch the model following the instructions in the repo's [README](../README.md).
3. Edit [./install-opencode.sh.j2](./install-opencode.sh.j2) with the right model name (e.g. substitute "moonshotai/Kimi-K2-Thinking").
4. Add your Swiss AI API key to the environment variables as `SWISS_AI_API_KEY`
5. Run the following (changing the model name as needed):
  ```sh
  harbor run -d terminal-bench@2.0 --agent-import-path swiss_ai_opencode:OpenCode -m swiss-ai/swiss-ai/Apertus-70B-Instruct-2509
  ```
