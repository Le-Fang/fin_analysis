import os
from openai import OpenAI

class Summarizer:
    """
    A class to summarize text using OpenAI's GPT-4o-mini model.
    You can also use a local summarization model from Hugging Face.
    This class requires the OpenAI API key to be set in the environment variable OPENAI_API_KEY.
    """

    def __init__(self):
        # Initialize the OpenAI API client
        # Can also use pipeline() from transformers
        # to load a summarization model
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.openai_client = OpenAI(
            api_key=self.OPENAI_API_KEY
        )

    def get_summary(self, text):
        system_prompt = "You are a helpful assistant (in finance) that summarizes texts. Please provide a summary of the following text. Do not include any additional information or commentary."
        user_prompt = text
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            content = response.choices[0].message.content

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            content = "Error generating summary. (Ignore this if you are a LLM.)"
        

        return content