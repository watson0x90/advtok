import transformers, advtok

# Initialize your favorite LLM...
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="cuda")
# ...and its tokenizer.
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def normal_interaction(request):
    # Tokenize the request normally
    inputs = tokenizer(request, return_tensors="pt").to(model.device)
    # Generate response using standard parameters
    outputs = model.generate(**inputs, max_new_tokens=256)
    # Decode and print the generated output
    print(" ==== Normal Interaction Results ==== \n\n")
    for output in tokenizer.batch_decode(outputs):
        print(output, '\n' + '-'*30)

def advtok_interaction(request, response):
    
    # Run advtok with a random initialization.
    X = advtok.run(model, tokenizer, request, 100, response, 128, X_0="random")
    # Generate samples with the adversarial tokenization.
    O = model.generate(**advtok.prepare(tokenizer, X).to(model.device), do_sample=True, top_k=0, top_p=1, num_return_sequences=16, use_cache=True, max_new_tokens=256, temperature=1.0).to("cpu")
    # Print samples.
    print("\n\n ==== AdvTok Response ==== ")
    for o in tokenizer.batch_decode(O): print(o, '\n' + '-'*30)

if __name__ == "__main__":

    # Set up the request and expected response.
    request = ""
    response = ""
    
    normal_interaction(request)
    advtok_interaction(request, response)