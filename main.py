import os
import torch
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
from subset_maker import convert_to_FAISS
# Parameters for Bits & Bytes enabling efficient training
# ---------------------------------------------------------
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Set up quantization config
# ---------------------------------
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Model config
# ---------------------------------
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
config_model = AutoConfig.from_pretrained(model, quantization_config=bnb_config)

# Base Case: Testing knowledge of programming prior to finetune
base_input = tokenizer.encode_plus("[INST] write a python function to loop through a list of numbers and print the "
                                   "output [/INST]",
                                   return_tensors="pt")['input_ids'].to('cuda')

generated_ids = model.generate(base_input,
                               max_new_tokens=1000,
                               do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(base_input)
print(decoded)


# Operational Functions
# -------------------------------
def Inference_pipeline(I_model: AutoModelForCausalLM, I_tokenizer: tokenizer, input: str,
                       task: str = "text-generation", return_text: bool = True):
    encoded_input = I_tokenizer.encode(input, return_tensors="pt")
    # noinspection PyUnresolvedReferences
    encoded_output = I_model.generate(encoded_input, max_length=100, temperature=0.5, repetition_penalty=1.1,
                                      return_full_text=return_text)
    decoded_output = tokenizer.decode(encoded_output[0], skip_special_tokens=True)
    return decoded_output


prompt_template = """
### [INST] 
Instruction: Answer the question based on your 
programming knowledge. Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
articles = ["https://www.w3schools.com/python/gloss_python_for_range.asp",
            "https://www.w3schools.com/python/python_for_loops.asp",
            "https://www.w3schools.com/python/python_functions.asp",
            "https://www.w3schools.com/python/python_lambda.asp"]

llm_chain = LLMChain(llm=Inference_pipeline, prompt=prompt)

# Creation of the RAG system in langchain
# ---------------------------------------
database, retriever = convert_to_FAISS(articles=articles, chunk_size=100, similarity=5)
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | llm_chain)

# call the rag_chain
# ---------------------------------------
rag_chain.invoke("write a python function to loop through a list of numbers and print the output")
