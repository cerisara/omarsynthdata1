# %%
!pip install pandas
!pip install sentence-transformers
!pip install accelerate
!pip install -U bitsandbytes
!pip install datasets

# %%
import pandas as pd
df = pd.read_csv('DATASET3/final.csv')
laliste=[]
laliste_labels=[]
for elt in df['label']:
    laliste_labels.append(elt)
for elt in df['text']:
    laliste.append(elt)

# %%
laliste_labels=[]
for elt in train['intent']:
    laliste_labels.append(elt)

# %%
from sentence_transformers import SentenceTransformer
import torch

# Define arguments for 4-bit quantization
model_kwargs = {
    "device_map": "auto",         # Automatically map layers to devices (GPU/CPU)
    "torch_dtype": torch.bfloat16,  # Recommended dtype for Qwen models
    "load_in_4bit": True,           # Enable 4-bit quantization
}

# The model card also recommends enabling flash_attention_2 for better performance
# You may need to install it: pip install flash-attn
# Set use_flash_attention_2=True if available
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs=model_kwargs,
    trust_remote_code=True, # Often required for custom model architectures
    # use_flash_attention_2=True, # Uncomment if flash-attn is installed
)

# %%
from datasets import load_dataset
ds = load_dataset("hrithikpiyush/acl-arc")
ds.set_format("pandas")
train=ds["train"][:]
validation=ds["validation"][:]
test=ds["test"][:]

# %%
documents = laliste[:10]

len(documents)

# %%
documents = laliste

synth_data = torch.from_numpy(embedding_model.encode(documents))


# %%


#synth_data = torch.from_numpy(embedding_model.encode(train['cleaned_cite_text'].tolist()))
validation_data = torch.from_numpy(embedding_model.encode(validation['cleaned_cite_text'].tolist()))
test_data = torch.from_numpy(embedding_model.encode(test['cleaned_cite_text'].tolist()))


# save synthdata into a file
torch.save(synth_data, 'synthdata1/synthdata.pt')
torch.save(validation_data, 'synthdata1/validation.pt')
torch.save(test_data, 'synthdata1/test.pt')

