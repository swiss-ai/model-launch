# How to run OpenCode

## 1. On Clariden

1. Check if the desired model is already running on the [API](https://serving.swissai.cscs.ch/.
2. If it is not, launch the model following the instructions in the [README](README.md).

## 2. On your dev machine (your laptop, Clariden, your group's server...)

1. Install [OpenCode](https://opencode.ai/docs/#install).
2. Once the model is live on the [API](https://serving.swissai.cscs.ch/), add the following to `~/.config/opencode/opencode.json` (or the desired [location](https://opencode.ai/docs/config/#locations)):
  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "swissai": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Swiss AI",
        "options": {
          "baseURL": "https://api.swissai.cscs.ch/v1",
          "apiKey": "{env:CSCS_API_KEY}"
        },
        "models": {
          "<value from --served-model-name>": {
            "name": "<desired-model-name-for-ui>"
          }
        }
      }
    }
  }
  ```
  Substitue `<value from --served-model-name>` and `<desired-model-name-for-ui>` accordingly.
4. Add your API key from the [API platform](https://serving.swissai.cscs.ch/api_key) to your env variables as `CSCS_API_KEY`.
3. Run `opencode` in the terminal, and run `/model` to select the model under the `<desired-model-name-for-ui>` name. You should be able to use it!
