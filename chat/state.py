import os
import reflex as rx
from openai import OpenAI
from chat.engine import Engine
import json


# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    password: str = 'sqWireT345$#-sefdSDsn'

    # The name of the new chat.
    new_chat_name: str = ""

    calculation_results: str

    latest_report: str = ""


    states = {
        'info_gathering': 'the state where the agent is gathering information from the user on the weights and sizes of goods to be shipped', 
        'calculating': 'the state where the agent is calculating the most optimal distribution of packaging according to information from packaging guides and Schedule B', 
        'report_generation': 'the state where the agent generates a report for the user based on the calculating results', 
        'engine_call': 'the state where the agent generates a json file that calls on the quote engine based on the calculating results'}


    prompts = {
        'info_gathering': '''\
    Your job is to take the user's requirements on what items they want to ship, ask them helpful \
    questions that will help you determine several criteria. These criteria are: \
    1. Sizes, 2. Weights. If the user doesn't know the exact weights or sizes of their goods, \
    you should estimate the weights and sizes, and ask the user if that looks right. \
    Your questions should not be overwhelming, do your best to group the goods together and ask about groups, \
    as well as first only ask about size information and then move on to weight information. Once you are done, \
    ask the user to wait while you fetch product and packaging info. Ask only one question at a time. In the end you need to find out the following \
    overall quantities: shipping by air or not, weight in lbs, value in $, distance to be shipped in miles.
    ''',
        'calculating': '''\
    Your job is to take dimensions provided in conversation by the user, take typical sizes of boxes and pallets, \
    and calculate a distribution of the user's goods into packaging while taking into account government regulations provided to you in the \
    materials. You should be detailed about the mathematics and sizing of it all, write out the math and detailed description of \
    the layount of goods in boxes and pallets if you need to. 
    In the end you need to find out the following \
    overall quantities: shipping by air or not, total weight in lbs, total value in $, total boxes number, total pallets number, distance to be shipped in miles
        ''',
        'report_generation': '''
        Your job is to generate a report for the user based on the calculations and their results provided to you. The report should detail how goods \
        must be packaged, what boxes and pallets they need to be packaged into, and detail government regulations that are relevant to the goods the user \
        needs to ship. At the very end you should ask the user if they would like to get quotes based on the report.  
        In the end you need to summarize the following quantities: 
        shipping by air or not, total weight in lbs, total value in $, total boxes number, total pallets number, distance to be shipped in miles
        ''',
        'engine_call': '''
        Your job is to generate a json based on a detailed packaging report, with the following fields:
        estimated_value, weight, box_number, pallet_number 
        '''
    }


    def get_output_engine(self, state):
        if state == 'info_gathering':
            return self.openai_process_question
        elif state == 'calculating':
            return self.calc_process_question
        elif state == 'report_generation':
            return self.openai_process_question
        elif state == 'engine_call':
            return self.engine_process_question
        else:
            raise ValueError(f"Invalid state: {state}")

    def get_prompt(self, state):
        return self.prompts[state]

    def info_materials(self, question):
        return dict()

    def calc_materials(self, question):
        with open('Freight Packaging (1).txt', 'r', encoding="utf8") as file:
            return {'Packaging Info': file.read()}
    
    def report_materials(self, question):
        with open('Freight Packaging (1).txt', 'r', encoding="utf8") as file:
            return {'Packaging Info': file.read()}

    def engine_materials(self, question):
        return None
    def update_state(self, question):
        prompt: str
        with open('State Decision Tree.txt', 'r', encoding="utf8") as file:
            prompt = file.read()

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            temperature=0,
            stream=False,
        )

        answer_text = session.choices[0].message.content
        print('====')
        print(answer_text)
        print('====')

        for key in self.prompts:
            if key in answer_text:
                return key
        return "STATE UPDATE ERROR"

    
    def get_materials(self, state, question):
        if state == 'info_gathering':
            return self.info_materials(question)
        elif state == 'calculating':
            return self.calc_materials(question)
        elif state == 'report_generation':
            return self.report_materials(question)
        elif state == 'engine_call':
            return self.engine_materials(question)
        else:
            raise ValueError(f"Invalid state: {state}")

    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):

            
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        if self.current_chat!=self.password:
            self.chats[self.current_chat][-1].answer= "Please Open Chat with Passcode Name"
            yield
            return

        chat_state = self.update_state(question)
        print(chat_state)
        if chat_state == "STATE UPDATE ERROR":
            self.chats[self.current_chat][-1].answer= "Try Again"
            yield
            return
        
        materials = self.get_materials(chat_state, question)
        prompt = self.get_prompt(chat_state)

        model = self.get_output_engine(chat_state)

        async for value in model(question, prompt, materials):
            yield value

        if chat_state == 'report_generation':
            self.latest_report = self.chats[self.current_chat][-1].answer

    async def openai_process_question(self, question: str, prompt: str, materials: dict):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ] + [{
                "role": "system",
                "content": f"{key_}: \n\n\n {val_}"
            } for key_, val_ in materials.items()] + [
                {"role": 'system',
                'content': f'Calculations So Far: {self.calculation_results}'}
            ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            stream=True,
        )

        # Stream the results, yielding after every word.
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer_text = item.choices[0].delta.content
                # Ensure answer_text is not None before concatenation
                if answer_text is not None:
                    self.chats[self.current_chat][-1].answer += answer_text
                else:
                    # Handle the case where answer_text is None, perhaps log it or assign a default value
                    # For example, assigning an empty string if answer_text is None
                    answer_text = ""
                    self.chats[self.current_chat][-1].answer += answer_text
                self.chats = self.chats
                yield

        # Toggle the processing flag.
        self.processing = False


    async def calc_process_question(self, question: str, prompt: str, materials: dict):
        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ] + [{
                "role": "system",
                "content": f"{key_}: \n\n\n {val_}"
            } for key_, val_ in materials.items()]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            stream=False
        )

        calculations = session.choices[0].message.content

        self.calculation_results += f'\n\n\n Latest Calculations: {calculations}\n\n'

        self.chats[self.current_chat][-1].answer = 'Calculations done, would you like to generate a packaging report?'
        yield

        # Toggle the processing flag.
        self.processing = False


    async def engine_process_question(self, question: str, prompt: str, materials: dict):
        self.chats[self.current_chat][-1].answer = "Generating Quotes..."
        yield
        self.chats[self.current_chat][-1].answer = ""
        quote_engine = Engine()
        prompt: str
        with open('JSONify.txt', 'r', encoding="utf8") as file:
            prompt = file.read()

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ]+ [
                {
                "role": "system",
                "content": f'Latest Report Summary of All Relevant Information: {self.latest_report}',
            },
        ]

        repeat_loop = True
        quotes: str
        while (repeat_loop):
            session = OpenAI().chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                temperature=0,
                stream=False,
            )

            json_string_output = session.choices[0].message.content

            print(json_string_output)

            quotes = quote_engine.get_quotes(json_string_output)

            repeat_loop = (quotes == -1)

        quotes_list = json.loads(quotes)

        for quote in quotes_list:
            self.chats[self.current_chat][-1].answer += f"{quote['company_name']}: {quote['quote']}\\$\n"
            yield