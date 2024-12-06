# Finetune-Llama2-dietCoach

**lama2-dietCoach** is a fine-tuned version of **LLaMA-2**, specifically tailored for providing diet recommendations. This model has been fine-tuned on a dedicated diet recommendation dataset for the task of text generation, leveraging **PEFT** (Parameter-Efficient Fine-Tuning) and **QLoRA** (Quantized Low-Rank Adaptation).

The complete fine-tuning process was executed on Google Colab, and the resulting model is hosted on the Hugging Face Model Hub for ease of access and deployment.

**Features**
**Diet Recommendation Expertise:** Generates personalized and context-aware diet recommendations.
**Efficient Fine-Tuning:** Utilizes PEFT and QLoRA to optimize resource usage during fine-tuning.
**Deployable Model:** Hosted on Hugging Face for seamless integration into your applications.

# Model Details
**Base Model:** LLaMA-2
**Fine-Tuning Frameworks:** PEFT and QLoRA
**Dataset:** Diet recommendation dataset (custom-curated)
**Hosting:** Hugging Face Model Hub (Link to the model: yousaf121/lama_dietCoach)

Installation
Clone the repository and install the dependencies:

git clone https://github.com/Myusuf121/lama2-dietCoach.git  
cd lama2-dietCoach  
pip install -r requirements.txt  

**Usage**
**Load the Model**
You can load the fine-tuned model directly from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "yousaf121/lama_dietCoach"  # Replace with your Hugging Face model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
input_text = "Provide a diet plan for weight loss for a 30-year-old person."
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Fine-Tuning Process**
**Dataset Preparation:** A diet recommendation dataset was curated and preprocessed for training.
**Notebook Execution:** The fine-tuning was conducted using a Colab notebook for efficient cloud-based computation.
**Model Upload:** The fine-tuned model was saved and uploaded to Hugging Face for accessibility.
For details on the fine-tuning process, refer to the provided notebook in this repository.

**Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the model or add new features.

**License**
This project is licensed under the MIT License.

**Acknowledgments**
Meta AI: For developing the LLaMA-2 base model.
Hugging Face: For providing tools and hosting the fine-tuned model.
Google Colab: For facilitating the training process.
