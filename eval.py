citation_dict={
        0:"BACKGROUND",
        1:"USES",
        2:"COMPARES OR CONTRASTS",
        3:"MOTIVATION",
        4:"CONTINUATION",
        5:"FUTURE"}

def prepval():
    from datasets import load_dataset
    from transformers import pipeline

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedder = pipeline("feature-extraction", model=model_name, trust_remote_code=True, device=-1)

    ds = load_dataset("hrithikpiyush/acl-arc")
    val = ds['validation']
    print("val",val)
    with open("aclarcval.lab","w") as g:
        for i in range(len(val)):
            l=val['intent'][i]
            g.write(str(l)+'\n')
    with open("aclarcval.txt","w") as g:
        for i in range(len(val)):
            l=val['cleaned_cite_text'][i]
            embeddings = embedder(l)
            g.write(str(embeddings)+'\n')
            g.flush()

prepval()
exit()


mlp = nn.Sequential(
    nn.Linear(dim, 256),
    nn.ReLU(),              # or nn.GELU() for Transformer-style
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, nclass)
)

for i in range(len(val)):


with torch.no_grad():
    validation_outputs = mymodel(validation_data.float())
    #test_outputs = mymodel(test_data)
    print(classification_report(validation['intent'], torch.argmax(validation_outputs, axis=1)))
