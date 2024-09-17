import chromadb

from models_pipeline.clients.phi3_client_w_openvino import Phi3Client 
# from langchain import PromptTemplate, LLMChain

from models_pipeline.vector_store import VectorStore
# from langchain.llms import YourLLMModel  # Replace with your actual LLM import

# # Initialize ChromaDB client
# client = chromadb.Client()
# collection = client.get_collection("your_collection_name")

# # Initialize your LLM
# # phi3_client = Phi3Client()
# # phi3_client.load_model()

# def get_relevant_info(query):
#     results = collection.query(query_texts=[query], n_results=5)
#     return "\n".join(results['documents'][0])

# def generate_question(student_query, context):
#     template = """
#     Based on the following information:
#     {context}
    
#     Generate a question related to the student's query: {query}
#     """
#     prompt = PromptTemplate(template=template, input_variables=["context", "query"])
#     llm_chain = LLMChain(prompt=prompt, llm=phi3_client.model)
#     return llm_chain.run(context=context, query=student_query)

# def analyze_answer(original_question, context, student_answer):
#     template = """
#     Original question: {question}
#     Relevant information: {context}
#     Student's answer: {answer}
    
#     Analyze the student's answer for correctness and completeness. Provide feedback.
#     """
#     prompt = PromptTemplate(template=template, input_variables=["question", "context", "answer"])
#     llm_chain = LLMChain(prompt=prompt, llm=phi3_client.model)
#     return llm_chain.run(question=original_question, context=context, answer=student_answer)



class Generation():
    def __init__(self):
        self.phi_client = Phi3Client()
        # phi_client.load_model()
        # self.input_processor = ProcessInput()
        self.phi_model = self.phi_client.load_model()
        self.model = self.phi_model.model
        self.tokenizer = self.phi_model.tokenizer
        self.storage = VectorStore() 

        #this class should be able to take information from the 
        # chromadb collection which should b updated with teh file 
        # that teh student put in and start the conversation  
        # so there should be a function that creates the initial start
        # of the conversation, and hten anpother funciton that analyzes 
        # student answer and determines if its correct or wrong, and then
        # give a follow up based on that
        # so there should be two types of follow ups -- correct and wrong ansewrs
        #  

    def initialize_conversation(self, topic):
        # Update the ChromaDB collection with the student's file
        # self.storage.get_or_create_collection(self, topic)

        # Retrieve relevant information from the vector store


        # context = self.storage.search_context()
        print(topic)
        context = self.storage.get_collection_info(topic)
        print("got collection info???")
        print(context)
        # Generate an initial prompt based on the context
        initial_prompt = self.generate_initial_prompt(context)
        print("generated initial prompt???")

        return initial_prompt
        

    def generate_initial_prompt(self, context):
        # Use the Phi model to generate an initial prompt
        print(context)
        # prompt = f"Based on the following context, generate an initial question or statement to start a conversation with a student:\n\n{context}"
        

        # prompt = f""" DO NOT OUTPUT THIS FOLLOWIZNG PROMPT IN YOUR RESPONSE, ONLY OUTPUT WHAT IS ASKED FOR. 
        # Based on the context at the end of the directions, and considering that you are Maestra, a teacher, you need to start a conversation with the user who is a student. 

        # You need to be able to start the conversation in a friendly manner such as the following:

        # Example 1: the student uploaded a file from their hw regarding World War 1
        # '<response_to_student> Hi! I'm Maestra, and it looks like you want to study about the World War 1 today. It can be hard to remember
        # everything you need to and this is definitely a hard topic, so let's get started! World war 1 was a battle between several countries including Germany, the US, and Britain. 
        # Do you know why they were fighting? <response_to_student>'

        # Example 2: the student uploaded a past test regarding the animal cell
        # '<response_to_student> Hi! I'm Maestra, and it looks like you want to study about the cell today. It can be hard to remember
        # everything you need to and this is definitely a hard topic, so let's get started! There's two types of cells, plant and animal according to your test. 
        # Can you tell me two differences between the two? <response_to_student>'

        # **Do not overburden the student with information or questions, only ask 1 question and wait for the user to respond. You do NOT need to give any note showing your resoning for 
        # your response to the user. Do NOT repeat the prompt in your response back, only give what I asked for here.** 
                    
        # Make sure to surround your initian of conversation to the student in these tags as shown in the examples, please include the final in the tags: <response_to_student_final> text </ response_to_student_final>
                    
        # Here is the context:
        # <beginning_of_context> 
        # \n\n{context}   
        # <end_of_context>
        # """

        prompt = f"""Based on the given context, generate a friendly conversation starter as Maestra, a teacher, addressing a student. Your response should be concise, ask only one question, and be enclosed in tags.

    Example format:
    <response_to_student_final>Hi! I'm Maestra. It looks like we're studying [topic] today. [Brief, friendly introduction]. [One question about the topic]?</response_to_student_final>

    Example 1: the student uploaded a file from their hw regarding World War 1
    '<response_to_student_final> Hi! I'm Maestra, and it looks like you want to study about the World War 1 today. It can be hard to remember everything you need to and this is definitely a hard topic, so let's get started! World war 1 was a battle between several countries including Germany, the US, and Britain. Do you know why they were fighting? </response_to_student_final>'
    
    **Do not overburden the student with information or questions, only ask 1 question and wait for the user to respond. You do NOT need to give any note showing your resoning for your response to the user. Do NOT repeat the prompt in your response back, only give what I asked for here.** 

    Make sure to surround your initian of conversation to the student in these tags as shown in the examples, please include the final answer in the tags: <response_to_student_final> text </response_to_student_final>
    
    Context:
    {context}

    The response to student:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=2000)
        initial_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return initial_prompt
    
    def check_answer(self, topic, your_question, student_answer, context):
        # Use the Phi model to generate an initial prompt
        print(student_answer)
        

        context = self.storage.get_collection_info(topic)
        print("got collection info???")

        prompt = f""" Based on the information in the context given and your previous question that the student tried to answer, determine if the studen't answer is correct.

        If the student answer is CORRECT, return a friendly response appreciating the student's resposne and followign up with another question. 
        Example: 
        "Nice answer! You're right, the fastest land animal on the planet is the cheetah! Do you know what makes it so fast?"

        If the student is INCORRECT, return a friendly response letting the student know that their answer is slightly incorrect, and give them the correct answer. Follow up with another question in relation to the context given.
        Example: 
        "Hm, I think your answer is almost correct, but not quite there yet. The fastest animal on the planet is not actually the lion. It's the cheetah! Do you know what makes it so fast?"

        Here's the necessary information: 

        The past question you had asked:
        {your_question}

        Student's answer: 
        {student_answer}

    Context:
    {context}

    The response to student:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=2000)
        initial_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return initial_prompt

    

    