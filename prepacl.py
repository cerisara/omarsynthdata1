def preparxiv():
    from datasets import load_dataset
    from transformers import pipeline
    import arxiv

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

    val = arxiv.loadArxiv()
    # ce corpus est deja split en phrases
    with open("arxivembeds.txt","w") as g:
        for i in range(len(val)):
            l=val[i]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()

def preptrain():
    from datasets import load_dataset
    from transformers import pipeline

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

    ds = load_dataset("hrithikpiyush/acl-arc")
    val = ds['train']
    print("data",val)
    with open("aclarctrain.lab","w") as g:
        for i in range(len(val)):
            l=val['intent'][i]
            g.write(str(l)+'\n')
    with open("aclarctrain.txt","w") as g:
        for i in range(len(val)):
            l=val['cleaned_cite_text'][i]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()
 
def preptest():
    from datasets import load_dataset
    from transformers import pipeline

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

    ds = load_dataset("hrithikpiyush/acl-arc")
    val = ds['test']
    print("data",val)
    with open("aclarctest.lab","w") as g:
        for i in range(len(val)):
            l=val['intent'][i]
            g.write(str(l)+'\n')
    with open("aclarctest.txt","w") as g:
        for i in range(len(val)):
            l=val['cleaned_cite_text'][i]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()
 
# preptest()
# preptrain()
preparxiv()

