from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Define the state: what data we’ll track
class State(TypedDict):
    question: str          # The input question
    answer: str = ""       # The generated answer
    category: str = ""     # The category (e.g., "science")
    tags: list = []        # List of tags with weights

# Define structured outputs using Pydantic
class Answer(BaseModel):
    answer: str = Field(description="Readable answer to the question")

class Category(BaseModel):
    category: str = Field(description="Single word category, e.g., 'science'")

class Tags(BaseModel):
    tags: list = Field(description="List of dicts with 'tag' and 'weight' (0-1)")

# Set up the language model (we’re using OpenAI’s GPT-4o)
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

# Node to generate an answer
def answer_node(state: State):
    prompt = f"Answer this question in a readable way: {state['question']}"
    response = llm.invoke(prompt).content
    return {"answer": response}

# Node to classify the question
def classify_node(state: State):
    prompt = f"Classify this question into a single word category: {state['question']}"
    response = llm.invoke(prompt).content
    return {"category": response}

# Node to generate tags

def tag_node(state: State):
    prompt = f"Generate 4 tags for this question with weights (0-1) showing importance, in JSON format: {state['question']}"
    response = llm.invoke(prompt).content
    import re
    import json
    cleaned_string = re.sub(r"```json|```", "", response).strip()
    parsed_json = json.loads(cleaned_string)
    return parsed_json

# def tag_node(state: State):
#     prompt = f"Generate 4 tags for this question with weights (0-1) showing importance, in JSON format. Return a JSON object with a 'tags' key containing an array of objects, where each object has a 'tag' key and a 'weight' key: {state['question']}"
#     response = llm.invoke(prompt).content
#     import json
#     try:
#         # Try to parse the response as JSON
#         parsed_json = json.loads(response)
#         # Check if the expected format is returned
#         if 'tags' in parsed_json:
#             return {"tags": parsed_json["tags"]}
#         else:
#             # If we got JSON but not in the expected format, transform it
#             transformed_tags = []
#             for tag_obj in parsed_json.get('tags', []):
#                 # Handle the case where tags are in format {"tag_name": weight}
#                 if isinstance(tag_obj, dict) and len(tag_obj) == 1:
#                     for tag, weight in tag_obj.items():
#                         transformed_tags.append({"tag": tag, "weight": weight})
#                 # Handle the case where tags already have tag and weight keys
#                 elif isinstance(tag_obj, dict) and 'tag' in tag_obj and 'weight' in tag_obj:
#                     transformed_tags.append(tag_obj)
#             return {"tags": transformed_tags}
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}")
#         # Return empty tags if parsing fails
#         return {"tags": []}

# def tag_node(state: State):
#     prompt = f"""Generate 4 tags for this question with weights (0-1) showing importance.
#     Return ONLY a JSON object with a 'tags' key containing an array of objects, where each object has a 'tag' key and a 'weight' key.

#     Example format:
#     {{\n        "tags": [\n            {{"tag": "example", "weight": 0.8}},\n            {{"tag": "another", "weight": 0.6}}\n        ]\n    }}

#     Question: {state['question']}"""
    
#     response = llm.invoke(prompt).content
#     print(f"LLM Response: {response}")  # Debug print
    
#     # Clean up the response by removing any text before the JSON
#     json_start = response.find('{')
#     if json_start != -1:
#         response = response[json_start:]
    
#     import json
#     try:
#         # Try to parse the response as JSON
#         parsed_json = json.loads(response)
        
#         # Validate the structure
#         if not isinstance(parsed_json, dict) or 'tags' not in parsed_json:
#             print("Invalid JSON structure")
#             return {"tags": []}
            
#         # Validate each tag
#         valid_tags = []
#         for tag_obj in parsed_json.get('tags', []):
#             if isinstance(tag_obj, dict) and 'tag' in tag_obj and 'weight' in tag_obj:
#                 valid_tags.append(tag_obj)
        
#         return {"tags": valid_tags}
        
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}\nResponse was: {response}")
#         return {"tags": []}

def combine_node(state: State):
    # We need to return a dictionary with keys that match the State class attributes
    # In this case, we're not actually changing any state values, just passing them through
    # If we want to add a 'response' field, we would need to add it to the State class first
    return {
        # Return the existing state values
        "question": state["question"],
        "answer": state["answer"],
        "category": state["category"],
        "tags": state["tags"]
    }

# Create a simple sequential graph (not parallel) that will still work
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("answer_node", answer_node)
graph.add_node("classify_node", classify_node)
graph.add_node("tag_node", tag_node)
graph.add_node("combine", combine_node)

# Create a simple sequential flow
graph.add_edge("answer_node", "classify_node")
graph.add_edge("classify_node", "tag_node")
graph.add_edge("tag_node", "combine")

# Define entry and exit points
graph.set_entry_point("answer_node")
graph.set_finish_point("combine")

# Compile the graph into an app
app = graph.compile()  # Recompile after updating

result = app.invoke({"question": "What is photosynthesis?"})
print(result)

### Test Part 1
# state = {"question": "What is photosynthesis?"}
# print(answer_node(state))  # Outputs something like {"answer": "Photosynthesis is..."}
# print(classify_node(state))  # Outputs {"category": "science"}
# print(tag_node(state))  # Outputs {"tags": [{"tag": "plants", "weight": 0.9}, ...]}