# GBV-support-resources-chatbot

This is a real-time RAG-based conversation agent, utilizing an existing large language model to create a response in natural language. Specifically, we aim to create a chatbot that provides **gender-based violence (GBV) support resources** in **Metro Vancouver, BC** based on a local dataset. 

## Context
When Jae Eun Kim, one of the team members, worked as a victim support worker for women exposed to violence at a local charity, she felt that accessing relevant service information (e.g. as phone numbers and operating hours for mental health centres) was often too inaccessible for young individuals, especially when it required making phone calls. 

## Use Case
This chatbot can be used in real life to quickly direct young individuals exposed to gender-based violence to information about relevant organizations. In other words, the chatbot will be used for resource provision, a task to give a list of organization names, short service description, and contact information. 

# Pipeline
Currently, we are planning to use LLaMA, a publicly available open-source large language model (LLM) developed by Meta AI. Creating the chatbot will require a pipeline that allows the system to search its database for the most semantically similar values as the user’s question. 
1. Prepare the data in a format for text embedding.
2. Use an embedding model to convert data into numerical embeddings capturing semantic meaning.
3. Export and save these embeddings into a database.
4. When the user types in a question to the chatbot, their questions will be treated as input ‘features’.
   - The features will also be converted into text embeddings using the embedding model.
5. Finally, the system will search the database for the most semantically similar values.
6. As output, LLaMA will generate a natural response. 

# Testing
Since this is a chatbot, we can test this model by directly typing into the chatbot. We can prepare several prompts about gender-based violence support services, including legal help, counselling, housing, and child support. For example, if a user asks for a ‘place to stay overnight in Burnaby’, the example output could be ‘If you’re in immediate danger, please call 911 or VictimLinkBC at 1-800-563-0808. In Burnaby, you can contact Ishtar Women’s Resource Society at 604-936-8710 or Dixon Transition Society at 604-298-3454 for 24-hour shelter and counselling support.”
