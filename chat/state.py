import os
import reflex as rx
from openai import OpenAI
from chat.engine import Engine
from chat.components.DataTable import DataTable
import json
import base64
import ast


# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
    imgs: list[str]


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

    uploading: bool = False

    password: str = 'sqWireT345$#-sefdSDsn'

    # The name of the new chat.
    new_chat_name: str = ""

    calculation_results: str

    latest_report: str = ""

    cols: list[dict] = [
        {
            "title": "Name",
            "type": "str",
            "width": 130,
        },
        {
            "title": "Weight",
            "type": "float",
            "width": 65,
        },
        {
            "title": "Length",
            "type": "float",
            "width": 65,
        },
        {
            "title": "Width",
            "type": "float",
            "width": 65,
        },
        {
            "title": "Depth",
            "type": "float",
            "width": 65,
        },
        {
            "title": "Quantity",
            "type": "int",
            "width": 65,
        },
    ]
    data = []
    change_logs = []

    def apply_update(self, update_list: list):
        self.change_logs.append(update_list + ['Updated By Assistant'])
        for i, row in enumerate(self.data):
            if row[0] == update_list[0]:
                for j in range(len(row)):
                    if not (update_list[j] is None):
                        self.data[i][j] = update_list[j]
                if update_list[-1] == -1:
                    self.data.pop(i)

                return
        if update_list[-1] != -1:
            self.data.append(update_list)

    def get_edited_data(self, pos, val):
        col, row = pos
        self.data[row][col] = val["data"]
        update_list = [self.data[row][0], None,None,None,None,None]
        update_list[col] = val["data"]
        self.change_logs.append(update_list + ['Updated By User'])


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
    you should estimate the weights and sizes, and ask the user if that looks right. The weights should be all in pounds, and the  \
    sizes should be in inches. You need to estimate the size of each good in three dimensions - Length, Width, and Depth. Clearly \
    label each dimension.

    Your questions should not be overwhelming, do your best to group the goods together and ask about groups, \
    as well as first only ask about size information and then move on to weight information. Once you are done, \
    ask the user to wait while you fetch product and packaging info. Ask only one question at a time. In the end you need to find out the following \
    overall quantities: shipping by air or not, weight in lbs, value in $, distance to be shipped in miles, day range for items to be shipped.

    LANGUAGE AND STYLING: you should be succinct and try not to overwhelm the user or repeat yourself unnecessarily. You should put in BOLD every \
    question that you ask the user. Make sure you don't ask too many questions in a single message. \
    When writing lists, write them in a compact style, so they are easily readable at a glance. Rather than using headings in lists, write \
    about each individual item in line and put those items in a list. When you write a list, never make it nested. \
    Instead of nested lists always use inline or just don't use numbered lists or bullet point lists at all. They look bad.

    When asking about shipping by air, advise the user that air shipping is faster but it is also more expensive.
    ''',
        'calculating': '''\
    Your job is to take dimensions provided in conversation by the user and in a data table summarizing the goods' sizes and weights. \
    The information in the table has precedence over the information in the conversation, if the two don't match. \
    The table has the following columns: 
    {
            "title": "Name",
            "type": "str",
        },
        {
            "title": "Weight",
            "type": "float",
        },
        {
            "title": "Length",
            "type": "float",
        },
        {
            "title": "Width",
            "type": "float",
        },
        {
            "title": "Depth",
            "type": "float",
        },
        {
            "title": "Quantity",
            "type": "int",
        }, 
    When calculating the packaging distribution, take typical sizes of boxes and pallets, \
    and calculate a distribution of the user's goods into packaging while taking into account government regulations provided to you in the \
    materials. You should be detailed about the mathematics and sizing of it all, write out the math and detailed description of \
    the layount of goods in boxes and pallets if you need to. 
    CONSTRAINTS: If at all possible you should use standard box sizes provided to you. Only use custom box/crate/pallet sizes if absolutely necessary.
    In the end you need to find out the following \
    overall quantities: shipping by air or not, total weight in lbs, total value in $, total boxes number, total pallets number, distance to be shipped in miles
        ''',
        'report_generation': '''
        Your job is to generate a report for the user based on the calculations and their results provided to you. The report should detail how goods \
        must be packaged, what boxes and pallets they need to be packaged into, and detail government regulations that are relevant to the goods the user \
        needs to ship. 

        Make sure that your report contains all information about what boxes and pallets must be used, and how many of each.

        LANGUAGE AND STYLING: You should be succinct and not add any unnecessary details. Make sure your report is as easy and unimposing to read \
        for the user as possible. Be Short and to the point, no need to include redundant information. The user only needs a summary of \
        quantities like weight, value etc... and some packaging guidelines. Keep it short. When you write a list, never make it nested. \
        Instead of nested lists always use inline or just don't use numbered lists or bullet point lists at all. They look bad.

        In the end you need to summarize the following quantities: 
        shipping by air or not, total weight in lbs, total value in $, total boxes number, total pallets number, distance to be shipped in miles
        ''',
        'engine_call': '''
        Your job is to generate a json based on a detailed packaging report, with the following fields:
        estimated_value, weight, box_number, pallet_number 
        '''
    }

    img: list[str]

    async def handle_upload(self, files: list[rx.UploadFile] = []):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        self.uploading = True
        for file in files:
            upload_data = await file.read()

            # Update the img var.
            self.img.append(base64.b64encode(upload_data).decode('utf-8'))
        self.uploading = False

    def get_output_engine(self, state):
        if 'info_gathering' in state:
            return self.openai_process_question
        elif 'calculating' in state:
            return self.calc_process_question
        elif 'report_generation' in state:
            return self.openai_process_question
        elif 'engine_call' in state:
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
        qa = QA(question=question, answer="", imgs=[])
        self.chats[self.current_chat].append(qa)

        new_q = question
        if question is None:
            new_q = ""
        if len(self.img) > 0:
            new_q += "\n\nSee Attached Images"


        self.chats[self.current_chat][-1].question = new_q
        self.chats[self.current_chat][-1].imgs = self.img

        self.img=[]

        if self.current_chat!=self.password:
            self.chats[self.current_chat][-1].answer= "Please Open Chat with Passcode Name"
            yield
            return

        chat_state = 'info_gathering'
        
        materials = self.get_materials(chat_state, question)
        prompt = self.get_prompt(chat_state)

        model = self.get_output_engine(chat_state)

        async for value in model(question, prompt, materials):
            yield value

        if chat_state == 'report_generation':
            self.latest_report = self.chats[self.current_chat][-1].answer

        table_updates = self.update_table()
        for up in table_updates:
            self.apply_update(up)
            yield

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
            for im in qa.imgs:
                messages.append(
                    {
                    "role": "user",
                    "content": [
                            {
                            "type": "text",
                            "text": "Identify all the items in this picture that the user may want to ship, identify a likely size for each item based on average sizes of similar items and how that item looks, and finally ask the user to confirm your estimates."
                            },
                        
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{im}"
                                }
                            }
                        ]
                    }
                )
            
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        messages.append({"role":"system", "content": f"Table containing latest information vetted by the user about the goods to be shipped: Columns: [Name, Weight, Length, Wifth, Depth, Quantity], Data: {self.data}"})


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
        self.chats[self.current_chat][-1].answer = 'Calculating...'
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


        messages.append({'role':'system', 'content': f"Data Table summarizing the goods' sizes and weights to calculate packaging materials: {question}"})



        # Start a new session to answer the question.
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            stream=False
        )

        calculations = session.choices[0].message.content

        self.calculation_results += f'\n\n\n Latest Calculations: {calculations}\n\n'

        self.chats[self.current_chat][-1].answer += '\n\nCalculations done, generating packaging report...\n\n'
        yield

        # Toggle the processing flag.
        self.processing = False


    async def engine_process_question(self, question: str, prompt: str, materials: dict):
        self.chats[self.current_chat][-1].answer += "Generating Quotes...\n\n"
        yield
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
            self.chats[self.current_chat][-1].answer += f"\n{quote['company_name']}: {quote['quote']}\\$\n"
            yield



    def update_table(self):
        self.processing = True
        prompt: str
        with open('TableJSON.txt', 'r', encoding="utf8") as file:
            prompt = file.read()

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role":"system", "content": f"Table containing latest information vetted by the user about the goods to be shipped: Columns: [Name, Weight, Length, Wifth, Depth, Quantity], Data: {self.data}"},
            {"role":"system", "content": f"Changelogs of the table so far: {self.change_logs}"}
        ]

        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})         
            messages.append({"role": "assistant", "content": qa.answer})

    

        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            temperature=0,
            stream=False,
        )

        list_string_output = session.choices[0].message.content

        print(list_string_output)

        updates_list = ast.literal_eval(list_string_output)
        self.processing = False
        return updates_list


    async def get_quotes(self):            
        qa = QA(question="", answer="", imgs=[])
        self.chats[self.current_chat].append(qa)

        if self.current_chat!=self.password:
            self.chats[self.current_chat][-1].answer= "Please Open Chat with Passcode Name"
            yield
            return

        async for val in self.calc_process_question(str(self.data), self.get_prompt('calculating'), self.get_materials('calculating', '')):
            yield val

        async for val in self.openai_process_question("", self.get_prompt('report_generation'), self.get_materials('report_generation', '')):
            yield val
        self.latest_report = self.chats[self.current_chat][-1].answer

        async for val in self.engine_process_question("", self.get_prompt('engine_call'), self.get_materials('engine_call', '')):
            yield val


