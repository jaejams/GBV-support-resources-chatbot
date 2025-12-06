import gradio as gr
import pandas as pd
import numpy as np
import re
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import os
import spaces

# --- Global/Cached Variables ---
try:
    # --- Load Data and Embeddings ---
    sheet_id = "1hMsYgDQj3ymqwxUXA7R-ITITnw3HzeVZBxaXAjiJwAE"
    sheet_gids = {
        "Starting Point": "0",
        "Immediate Help": "1278392561",
        "Counselling": "713986636",
        "Child/Youth Counselling": "1265113400",
        "Parenting": "299805447",
        "Safe Housing": "1571281149",
        "Victim Rights Info": "1952909822",
        "Legal Rep": "958128700",
        "Legal Info": "1989315755",
        "Grief": "2127423570"
    }

    all_dfs = []
    for sheet_name, gid in sheet_gids.items():
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        try:
            df = pd.read_csv(url)
            df['Source_Sheet'] = sheet_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {sheet_name}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['Combined Description'] = combined_df['Relevant crime/incident'].astype(str) + '; ' + combined_df['Description'].astype(str)
        print(f"DF COMBINED! {combined_df}")
    else:
        combined_df = pd.DataFrame()
        print("WARNING: Dataframe is empty.")

    # --- Load Embedding Model ---
    print("Loading Embedding Model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding Model LOADED!")     

    if not combined_df.empty:
        text_to_embed_description = combined_df['Combined Description'].fillna('').astype(str).tolist()
        embeddings_description = embedding_model.encode(text_to_embed_description)
        combined_df['embeddings_description'] = list(embeddings_description)
        print(f"DF UPDATED! {combined_df}")

    else:
        print("WARNING: Skipping embedding generation due to empty DataFrame.")

    HF_AUTH_TOKEN = os.environ.get("HF_TOKEN")
    
    print("Loading Llama Model...")
    model_name = "meta-llama/Llama-2-7b-chat-hf" 
    print(f"llama model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_AUTH_TOKEN)
    
    llm = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto", 
        torch_dtype=torch.float16,
    )
    
    print("LLAMA loaded!!")

except Exception as e:
    print(f"FATAL ERROR during model or data loading: {e}")


# --- Constants ---
DESC_THRESHOLD = 0.2 # this is for initializing the conversation. If the start of the conversation doesn't meet this, then the chatbot's keep asking for more questions. 
FINAL_THRESHOLD = 0.4 # this is for filtering out the most relevant information. 
N_DESC = 20

# --- Global Chat Context State ---
# These will be updated within chatbot_loop and persist across calls
LAST_KNOWN_INTENT = None
LAST_KNOWN_CITY = None
desc_results_df = None

SYSTEM_PROMPT = """
YOU ARE A TRAUMA-INFORMED, COMMUNITY-CONNECTED SUPPORT AGENT DESIGNED TO ASSIST INDIVIDUALS EXPERIENCING GENDER-BASED VIOLENCE IN BRITISH COLUMBIA, CANADA. 
"""
# Future plan: make this more relevant to LAST_KNOWN_INTENT 
SYSTEM_PROMPT_RAG = SYSTEM_PROMPT + """
\n
**RAG INSTRUCTIONS**: NEVER START YOUR RESPONSE WITH A GREETING. YOUR ONLY TASK IS TO PROVIDE A SUMMARY OF THE FOLLWING SERVICE INFORMATION, delimited by triple backticks (```), to formulate your response WITHIN 50 WORDS. Explain how the summary relevant to user input, which is delimited by triple exclamation marks (!!!). MAINTAIN WARM ATTITUDE, BUT SUMMARY SHOULD BE IN 50 WORDS. **Do not mention the RAG process, the triple backticks, or triple exclamation marks in your final answer**.
"""

CITY_KEYWORDS = {
    "new west": "New Westminster",
    "new westminster": "New Westminster",
    "surrey": "Surrey",
    "vancouver": "Vancouver",
    "downtown vancouver": "Vancouver",
    "richmond": "Richmond",
    "north van": "North Vancouver",
    "north vancouver": "North Vancouver",
    "burnaby": "Burnaby",
    "west van": "West Vancouver",
    "west vancouver": "West Vancouver",
    "langley": "Langley",
    "coquitlam": "Tri-Cities (Port Moody, Coquitlam, Port Coquitlam)",
    "port moody": "Tri-Cities (Port Moody, Coquitlam, Port Coquitlam)",
    "port coquitlam": "Tri-Cities (Port Moody, Coquitlam, Port Coquitlam)",
}

VALID_CITY_CATEGORIES = [
    "New Westminster",
    "Surrey",
    "Vancouver",
    "Richmond",
    "North Vancouver",
    "Burnaby",
    "West Vancouver",
    "Langley",
    "Delta",
    "White Rock",
    "Tri-Cities (Port Moody, Coquitlam, Port Coquitlam)",
    "Other cities in BC, Canada",
]


# --- Core RAG Functions ---
def retrieve_with_pandas_description(query, top_k=N_DESC):
    print(f"I'm at retrieve_with_pandas_desc with {query}")

    if combined_df.empty:
        return pd.DataFrame()
    query_embedding = embedding_model.encode([query])[0]
    combined_df['similarity_desc'] = combined_df['embeddings_description'].apply(lambda x: np.dot(query_embedding, x) /
                                             (np.linalg.norm(query_embedding) * np.linalg.norm(x)))
    results = combined_df.sort_values(by="similarity_desc", ascending=False).head(top_k).copy()
    return results

def is_query_only_cities(query):
    # Normalize the query by removing common delimiters and whitespace
    normalized_query = re.sub(r'[,\s]+', ' ', query.lower()).strip()
    
    if not normalized_query:
        return False

    # Check if the normalized query is an exact match for one of the CITY_KEYWORDS keys
    if normalized_query in CITY_KEYWORDS:
        return True

    # Check if the query is composed entirely of city keywords separated by spaces
    # This handles "surrey burnaby" or "new west"
    
    # Check for multi-word city keywords first (e.g., "new westminster")
    for keyword in sorted(CITY_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in normalized_query:
             # Remove the detected keyword from the string
            normalized_query = normalized_query.replace(keyword, '').strip()

    # After removing all city keywords, if the string is empty or contains only delimiters, 
    # it means the original query was only city names.
    if not normalized_query:
        return True
    
    # Fallback check (less rigorous, but helps)
    # Check if the remaining non-city parts contain any content words (excluding "and", "or", etc.)
    non_city_words = re.sub(r'\b(and|or|in)\b', '', normalized_query).strip()
    return not non_city_words

def remove_substrings_from_string(main_string, substrings_list):
    """
    Removes city names and optional preceding prepositions (like 'in', 'at', 'for')
    from the main string, case-insensitively, to isolate the intent.
    """
    cleaned_string = main_string
    print(f"[DEBUG] main_string '{cleaned_string}'")
    # 1. Define the prepositions we want to optionally remove
    prepositions = r'(?:\s*(?:in|at|for)\s+)?' # Matches optional ' in ', ' at ', ' for '
    
    # Use a set of canonical cities for efficiency
    canonical_cities = set(substrings_list)
    
    for canonical in canonical_cities: 
        # Find all keywords associated with this canonical city (e.g., 'new west', 'new westminster')
        keywords = [key for key, city in CITY_KEYWORDS.items() if city == canonical]
        
        # Sort keywords by length in descending order to match multi-word names first
        keywords.sort(key=len, reverse=True)
        
        for keyword in keywords:
            # Construct a robust regex pattern: [Prepositions]? [City Keyword]
            # The '(\s*|$)'' at the end handles cases where the city is at the end of the sentence
            pattern = rf'{prepositions}\b{re.escape(keyword)}\b(\s*|$)'
            
            # Use sub to replace the matched pattern (including the optional preposition) with a single space
            # flags=re.IGNORECASE ensures case-insensitive matching
            cleaned_string = re.sub(pattern, ' ', cleaned_string, flags=re.IGNORECASE)

    # 2. Clean up resulting extra spaces, commas, and strip leading/trailing whitespace
    cleaned_string = re.sub(r'[\s,]+', ' ', cleaned_string).strip()
    print(f"[DEBUG] cleaned_string '{cleaned_string}'")
    return cleaned_string

def get_df_filtered_by_desc(query):
    print(f"[DEBUG] get_df_filtered_by_desc with query: '{query}'")
    
    return retrieve_with_pandas_description(query, top_k=N_DESC)

def detect_city_from_query(query):
    print(f"detect_city_from_query with {query}")
    text = query.lower()
    detected = []
    for keyword, canonical in CITY_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword.lower())}\b", text):
            detected.append(canonical)
    print(f"found city: {detected}")
    return detected

def get_df_filtered_by_general_city(city_context, desc_results_df):
    print(f"[STEP] I'm at get_df_filtered_by_general_city with query: {city_context}.")
    print(f"[DEBUG] As of now desc_df is = {desc_results_df}")

    pattern = r'\b(?:' + '|'.join(re.escape(c) for c in city_context) + r')\b'
    general_city_filtered_df = desc_results_df[desc_results_df["City"].str.contains(pattern, case=False, na=False, regex=True)]
    print(f"[DEBUG] FILTERED by GENERAL city! General_city_filtered_df is = {general_city_filtered_df}")
    
    return city_context, general_city_filtered_df

def get_df_filtered_by_service_city(city_context, general_city_filtered_df):
    print(f"[STEP] I'm at get_df_filtered_by_SERVICE_city with: '{city_context}'")

    pattern = r'\b(?:' + '|'.join(re.escape(c) for c in city_context) + r')\b'
    main_city_filtered_df = general_city_filtered_df[general_city_filtered_df["Main Service City"].str.contains(pattern, case=False, na=False, regex=True)]
    print(f"[DEBUG] FILTERED by MAIN city! Now, main_city_filtered_df is = {main_city_filtered_df} THIS WILL FEED THE FINAL RESULTS!!")
    
    return main_city_filtered_df if not main_city_filtered_df.empty else general_city_filtered_df

@spaces.GPU(duration=250)
def llm_generate_response(prompt):
    prompt_template = f"<s>[INST] {prompt} [/INST]"
    try:
        response = llm(
            prompt_template,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = response[0]['generated_text']
        response_text = generated_text.split("[/INST]")[-1].strip()
        return response_text
    except Exception as e:
        print(f"LLM generation error: {e}")
        return "I encountered an error generating a response. Please try again."
    
    
def generate_resources(FINAL_FILTERED_DF):
    global LAST_KNOWN_INTENT
    global SYSTEM_PROMPT_RAG
    global FINAL_THRESHOLD
    
    # Filter rows above threshold — NEW DATAFRAME
    FINAL_FILTERED_DF = FINAL_FILTERED_DF[
        FINAL_FILTERED_DF["similarity_desc"] >= FINAL_THRESHOLD
    ]

    if FINAL_FILTERED_DF.empty:
        return (
            "Unforunately, no highly relevant services were found. "
            "However, please check out the below resource." # Introduce VictimLinkBC after this. Future implementation: get the rows from previous dataframes that were not included in FINAL_FILTERED_DF.
        )
    
    print(f"[STEP] I'm at generate_resources with df: {FINAL_FILTERED_DF}")
    
    
    context_list = []
    print(f"[STEP] I'm at generate_resources with df. Creating context_list!")
    for _, row in FINAL_FILTERED_DF.iterrows():
        
        phone = row['Phone #']
        email = row['Email']
        website = row['Website']
        
        # Use pd.isna() or np.isnan() to check for missing values (NaN)
        # If the value is missing, use 'N/A', otherwise use the value.
        phone_val = 'N/A' if pd.isna(phone) else phone
        email_val = 'N/A' if pd.isna(email) else email
        website_val = 'N/A' if pd.isna(website) else website
        
        combined_prompt = (
            f"{SYSTEM_PROMPT_RAG}\n"
            f"Generate a summary of this: ```\n{row['Description']}\n```. Explain how it is relevant to !!!{LAST_KNOWN_INTENT}!!!"
        )

        # Call LLM and store the result before building the final string
        print(f"[STEP] Fetched org name: {row['Title']}")
        print(f"Generating response for: {row['Title']}...")
        
        llm_summary = llm_generate_response(combined_prompt)
        
        print(f"finally got LLM summary! {llm_summary}...")
        
        context_entry = (
            f"Organization Name: {row['Title']}\n"
            f"{llm_summary}\n"
            f"📞: {phone_val}, 📧: {email_val}, 🌐: {website_val}\n"
            "--------------------------------------------------------------------------"
        )
        context_list.append(context_entry)
        print(f"[STEP] Completed processing for: {row['Title']}")
        # print(f"[STEP] I'm at generate_resources with CONTEXT_LIST []: {context_entry}")
        
    return "\n\n".join(context_list)

    
def chatbot_loop(query, history):
    global LAST_KNOWN_INTENT, LAST_KNOWN_CITY
    global desc_results_df

    city_context = detect_city_from_query(query)
    is_first_interaction = not history
    
    if is_query_only_cities(query):
        # the query itself indicates a city.
        print(f"query is a city name itself ('{query}') ")
        
        if LAST_KNOWN_INTENT and city_context:
            # LAST_KNOWN_CITY's been saved in the past. User previously gave intent, now they gave the city.
            LAST_KNOWN_CITY = city_context
            print(f"[CITY ONLY] We've LAST_KNOWN_INTENT: '{LAST_KNOWN_INTENT}'. New city input: '{LAST_KNOWN_CITY}' as city_context.")
        else:
            # User gave city first, or gave city again. Ask for intent.
            LAST_KNOWN_CITY = query
            print(f"[CITY ONLY] INTENT's !!!NOT!!! been saved in the past. Saving the query: '{LAST_KNOWN_CITY}' as LAST_KNOWN_CITY.")
            
            if is_first_interaction:
                return (
                "Hello, I am happy that you have found me. My name is One Tap Away, designed for gender-based violence support services resources."
                "\n Please note that my answer is only restricted to Metro Vancouver, BC!"
                "\n Thank you for letting me know the city. Which areas do you need help with? For example: counselling, safe housing, or legal information?"
                    )
            else:
                return "Thank you for letting me know the city. Which areas do you need help with? For example: counselling, safe housing, or legal information?" 
    
    elif not city_context:
        # the input's not city i.e. the query itself indicates an intent. 
        print(f"query is an intent itself ('{query}') ")
        
        if LAST_KNOWN_CITY:
            # LAST_KNOWN_CITY's been saved in the past.
            LAST_KNOWN_INTENT = query 
            print(f"[INTENT ONLY] City's been given in the past: '{LAST_KNOWN_CITY}', now the user's giving the intent! Saving the query: '{LAST_KNOWN_INTENT}' as INTENT.")
        
        else:
            # LAST_KNOWN_CITY's !!!NOT!! saved in the past.  
            LAST_KNOWN_INTENT = query
            print(f"[INTENT ONLY] no city_context with: {query}. Saving the query: '{LAST_KNOWN_INTENT}' as INTENT. This is potentially when the user just provided help areas without city info.")
            
            if is_first_interaction:
                return (
                "Hello, I am happy that you have found me. My name is One Tap Away, designed for gender-based violence support services resources."
                "\n Please note that my answer is only restricted to Metro Vancouver, BC!"
                "\n I would appreciate more details for your inquiries. Which city are you looking for services in? Vancouver, Surrey, Burnaby, Richmond, Langley, Coquitlam, Port Moody, Port Coquitlam, West Vancouver, North Vancouver, White Rock, Delta, Others?"
                    )
            else:
                return (
                    "Which city are you looking for services in? Vancouver, Surrey, Burnaby, Richmond, Langley, Coquitlam, Port Moody, Port Coquitlam, West Vancouver, North Vancouver, White Rock, Delta, Others?"
                    "\n ❗Please make sure to include ONLY the city name, WITHOUT any prepositions like 'in', or 'at'❗ "
                ) 
    else: 
        # user has provided both CITY_CONTEXT and INTENT at the same time. 
        print(f"[CITY & INTENT] City AND intent detected in one input!")
        LAST_KNOWN_CITY = city_context
        print(f"LAST_KNOWN_CITY IS SET!: '{LAST_KNOWN_CITY}'")
        LAST_KNOWN_INTENT = query
        print(f"LAST_KNOWN_INTENT IS SET! '{LAST_KNOWN_INTENT}'")
     
    # FINALLY! We got both city and intent.  
    if LAST_KNOWN_CITY and LAST_KNOWN_INTENT:
        LAST_KNOWN_INTENT = remove_substrings_from_string(LAST_KNOWN_INTENT, LAST_KNOWN_CITY) # if there's city info in the LAST_KNOWN_INTENT, then remove it.
        print(f"[CITY & INTENT] FINALLY we got LAST_KNOWN_CITY: '{LAST_KNOWN_CITY}' AND LAST_KNOWN_INTENT: '{LAST_KNOWN_INTENT}' ")
        
        desc_results_df = get_df_filtered_by_desc(LAST_KNOWN_INTENT)
        # RETRIEVAL: Print similarity scores
        print("--- Stage 1: Description Similarity Scores ---")
        print(desc_results_df[['Title', 'similarity_desc']].head())
        print(f"Max Similarity: {desc_results_df['similarity_desc'].max():.4f}")
        print(f"Min Similarity: {desc_results_df['similarity_desc'].min():.4f}\n")
        
        
        if desc_results_df.empty or desc_results_df.get('similarity_desc', pd.Series([-1])).max() < DESC_THRESHOLD:
            if is_first_interaction:
                return "Hello, I am happy that you have found me. My name is One Tap Away, designed for gender-based violence support services resources. \n Please note that my answer is only restricted to Metro Vancouver, BC! \n\n Could you please explain to me more about the area you need help with?"
            else:
                return "I am sorry, could you please elaborate more on what areas you'd like help with?"

        detected_cities, general_city_df = get_df_filtered_by_general_city(LAST_KNOWN_CITY, desc_results_df)
        final_df = get_df_filtered_by_service_city(detected_cities, general_city_df)

        final_statement = "Thank you for the information. Here is some relevant information for you:\n\n"
        final_output = generate_resources(final_df)
        final_resource = "\n\n I would like to highlight that VictimLink BC could be a good start for you. \n VictimLink BC is a toll-free, confidential and multilingual services available across B.C. and the Yukon. VictimLinkBC provides information and referral services to call victims of crime and immediate crisis support to victims of family and sexual violence, victims of human trafficking and sexual services. \n https://victimlinkbc.ca/ \n 1-800-563-0808 \n 211-victimlinkbc@uwbc.ca "
        
        LAST_KNOWN_INTENT = None
        LAST_KNOWN_CITY = None
        print(f"Wiping intent & city. Intent: '{LAST_KNOWN_INTENT}' City: '{LAST_KNOWN_CITY}'")

        return final_statement + final_output + final_resource # used to be llm_generate_response(combined_prompt)


def respond(message, history, system_message, max_tokens, temperature, top_p):
    
    llm_response = chatbot_loop(message, history)
    
    # 2. Yield the complete string once, and let the function end.
    yield llm_response
    # DO NOT use `return llm_response` after the yield.

chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a trauma-informed support agent for GBV in BC.", label="System message", visible=False),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens", visible=False),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature", visible=False),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p", visible=False),
    ],
    title="One Tap Away Chatbot",
    theme="soft",
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()

if __name__ == "__main__":
    demo.launch(share=True)
