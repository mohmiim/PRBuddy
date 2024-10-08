from langchain.output_parsers import ResponseSchema
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class SecurityReviewBuddy(object):
    LLAMA = False
    DEBUG = False

    def __init__(self):
        if self.LLAMA:
            self.chat = Ollama(model="llama3.2")
        else:
            self.chat = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', streaming=True)
        self.prompt = [
            ("system", """\
             Act as an AppSec Security expert doing a code review for given changes done to file, changes are provided in github patch format, and
             the file before the changes, review the changes and provide a review of the changes in Markdown format.
             Include the file name in the review as a markdown title on the top, provide only constructive feedback and 
             assume the code compiles fine. Give priority to OWSAP top 10 violations. Combine similar feedback points
             together in one point
             """),
            ("user", "{input}")
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(self.prompt)
        self.chain = self.prompt_template | self.chat | StrOutputParser()

    def convert(self, text):
        if self.DEBUG:
            print(self.prompt_template1)
        out = self.chain.invoke(text)
        if self.DEBUG:
            for key, value in out:
                print(key + " ==> " + str(value))
        return out
