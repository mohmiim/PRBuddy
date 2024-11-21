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
        self.prompt = [
            ("system", """\
             Act as a service given the changes done to a software project in github, it will return a description of 
             these changes in Markdown format to be used as a Pull Request description.
             The description should have a title of up to 6 words summarizing the changes, followed by up to 2 
             sections one tilted "Logic changes" and one titled "Test changes" the logic changes section will contain a 
             subsection for each file that is not a test file titled with the file name each of these sub sections will 
             contain a bullet list summarizing the important changes in this file and their reason do not include more than 3 bullets.
             The Test changes section will be included only if there are test files and its contents will follow the same format as the Logic Section.
             The description should be in Markdown format. do not include any thing in the response other than markdown.
             """),
            ("user", "{input}")
        ]

        template_str = """\
        describe the changes done to a software project in github, the description of 
             these changes should be in Markdown format to be used as a Pull Request description.
             The description should follow the provided description sample if it is not empty otherwise it should have a
             title of up to 6 words summarizing the changes, followed by up to 2 sections one tilted "Logic changes" and
             one titled "Test changes" the logic changes section will contain a subsection for each file that is not a 
             test file titled with the file name each of these sub sections will contain a bullet list summarizing the 
             important changes in this file and their reason do not include more than 3 bullets. The Test changes 
             section will be included only if there are test files and its contents will follow the same format as the 
             Logic Section. The description should be in Markdown format. do not include any thing in the response 
             other than markdown.
             The changes are: {change}
             Description Sample: {sample}
        """
        self.rephrase_template = PromptTemplate(input_variables=["change","sample"],
                                           template=template_str)

        self.prompt_template = ChatPromptTemplate.from_messages(self.prompt)
        self.chain = self.prompt_template | self.chat | StrOutputParser()

    def convert(self, text):
        if self.DEBUG:
            print(self.prompt_template1)
        #out = self.chain.invoke(text)
        formated_template = self.rephrase_template.format(change= text, sample = "")
        out = self.chat.invoke(formated_template)
        print(formated_template)
        if self.DEBUG:
            for key, value in out:
                print(key + " ==> " + str(value))
        return out
