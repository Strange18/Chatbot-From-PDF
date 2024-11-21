# Features of This Project (but not limited to)

User is able to Chat according to the pdf uploaded. \
A conversational form for collecting user information (Name, Phone Number, Email) \

Sending the email regarding the confirmation of booking through conversation

## Information about the setup
PDF setup -> I have tested the project with the annual report of Thapathali Campus, IOE that is found in [official website](https://tcioe.edu.np/). 

LLM Used -> [Gemini](https://ai.google.dev/gemini-api/docs/api-key) for embedding and response generation
and [Mistral AI](https://console.mistral.ai/) for User detail extraction and email content generation


Tools -> [Gmail Toolkits](https://python.langchain.com/docs/integrations/tools/gmail/)
 
## To Run the code locally

Add GEMINI API KEY and MISTRAL API KEY in the .env file with the key as \
\
GEMINI_API_KEY = 'cxdas...' \
MISTRAL_API_KEY = 'dasdsa...'
\
\
Mistral was used for detail extraction and email content generation. \
\
Also for the email sending feature get credentials.json from the developer console by following the tutorial [here](https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application)

### Install the packages

`pip install -r requirements.txt`

Then run

`streamlit run app.py`