from langchain.output_parsers import ResponseSchema
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class ChangeRephraser(object):
    LLAMA = True
    DEBUG = False

    def __init__(self):
        if self.LLAMA:
            self.chat = Ollama(model="llama3.2")
        else:
            self.chat = ChatOpenAI(temperature=0, model_name='gpt-4', streaming=True)

        template_str = """\
        describe the changes done to a software project in github, the description of 
             these changes should be in Markdown format to be used as a Pull Request description.
             The description should follow the provided description sample. Do not include any thing in the response 
             other than markdown.
             The changes are: {change}
             Description Sample: {sample}
        """
        self.rephrase_template = PromptTemplate(input_variables=["change", "sample"],
                                                template=template_str)

        # check if the file description_template exists
        # if it does, load it
        self.description_template = ""
        try:
            with open("description_template", "r") as f:
                self.description_template = f.read()
        except:
            pass

    def convert(self, text):
        if self.DEBUG:
            print(self.rephrase_template)
        # out = self.chain.invoke(text)
        formatted_template = self.rephrase_template.format(change=text, sample=self.description_template)
        if self.DEBUG:
            print(formatted_template)
        out = self.chat.invoke(formatted_template)
        if self.DEBUG:
            for key, value in out:
                print(key + " ==> " + str(value))
        # check if out is a String
        if isinstance(out, str):
            return out
        return out.content
