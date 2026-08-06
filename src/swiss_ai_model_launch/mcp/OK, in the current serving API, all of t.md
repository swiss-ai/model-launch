OK, in the current serving API, all of the models are launched and served for all of the users, which is not very good. We want to separate the models for the users.

As you know, We use SML or k8s to launch a model, via OpenTela, and each model in OpenTela may has a few labels. We want to introduce a new label for "authorization" to indicate which users are allowed to access a specific model. In this way, once user is launching a model via SML, they can specify who will be authotized to use the model. The options should be as below.
- `sml ... --authorization <user1@epfl.ch,user2@ethz.ch>`: This option allows the user to specify a comma-separated list of email addresses of users who are authorized to access the model. Only the specified users will be able to use the model once it is launched.
- `sml ... --authorization public`: This option allows the user to make the model accessible to all users. Any user can access and use the model once it is launched.
- `sml ... --authorization private`: This option allows the user to restrict access to the model to only themselves. No other users will be able to access or use the model once it is launched.

Default should be private.

Then, sml should pass these authorization labels to OpenTela when launching the model. OpenTela will then use these labels to enforce access control, ensuring that only authorized users can access the model.

Note that private means nothing to the Serving API as it don't know who has launched the model, so, SML, should translate it before hand. SML already has access to Serving API API key (CSCS API Key). Serving API must expose an an endpoint for whoami so that once it is hitted within the API key, it returns the email address of the user associated with that API key. This way, when a user specifies `--authorization private`, SML can call the Serving API's whoami endpoint to retrieve the user's email address and then pass that email address as the authorization label to OpenTela.

Then, in serving API, two things have to be done:
1. In frontend, we should only show public models and the models that the user is authorized to access based on the authorization labels. This means that when a user queries for available models, the Serving API should filter the models based on the user's email address and the authorization labels associated with each model. If the user was not logged in, they should only see public models.
2. In backend, we should enforce access control based on the authorization labels. When a user tries to access a model, the Serving API should check the authorization labels associated with that model and verify if the user's email address is included in the list of authorized users. If the user is not authorized, the Serving API should return an appropriate error message indicating that access is denied.

Here are the list of relevant repos:

- /users/ahadinia/repositories/model-launch
- /users/ahadinia/repositories/serving-api
