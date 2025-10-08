from transformers import pipeline

model_name = "Qwen/Qwen3-Embedding-0.6B"

# from transformers import AutoModel, AutoTokenizer
# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
# 
# txt=["aa bb cc", "dd ee ff"]
# inputs = tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
# 
# with torch.no_grad():
#     outputs = model(**inputs)
# 
# embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling example

embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

with open("ds6.txt","w") as g:
    with open("shuffledds6.csv","r") as f:
        for l in f:
            l=l[2:]
            if l[0]=='"': l=l[1:-1]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()

