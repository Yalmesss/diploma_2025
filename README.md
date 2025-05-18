This work explores the application of LLMs for the classification of user health messages, focusing on extracting structured medical information from unstructured text. The work combines both real and synthetically generated health messages to enhance model performance and generalization.
Continued pretraining was conducted on three open-source models -- Mistral, DeepSeek, and Gemma  using domain-specific medical texts and the DoRA (Diagonal Low-Rank Adaptation) approach to efficiently inject domain knowledge. These models were subsequently fine-tuned to extract structured outputs from user messages in a predefined JSON format, including symptoms, medical category, diagnosis, recommendations, suggested medications (with analogs), and generalized summaries of the input.
In addition to extraction, the reasoning version of DeepSeek model was used to generate detailed, human-readable medical reports based on the output of the classification model. This second-stage generation relied on a few-shot learning approach.
The models were evaluated based on their ability to accurately classify and extract relevant medical information, understand the medical context, adhere to the output format, and generate relevant responses. The best-performing classification model and the DeepSeek-based generation model were integrated into a web-based interface, allowing for real-time user interaction. 

Structure of the repository:

1. Folder 'Code'
   1.1 Folder 'deepseek' -  All that was used for two step fine tuning and evaluating DeepSeek-7b model
       1.1.1 deepseek_pretraining.py - pretraining the DeepSeek-7b model
       1.1.2 eval_deepseek.py - evaluating the quality of classification and structured output after full finetuning DeepSeek model
       1.1.3 instruct_tuning_deepseek.py - instrucion tuning of DeepSeek-7b model
   
   1.2 Folder 'deepseek_r1' - All that was used for few shot learning of the model DeepSeek-R1 model
       1.2.1 few_shot_deepseek.py - few shot learning of the model DeepSeek-R1 model
       1.2.2 test_deepseek_r1.py - evaluation of the generated outputs

   1.3 Folder 'gemma' - All that was used for two step fine tuning and evaluating Gemma-7b model
       1.3.1 instruct_tuning_gemma.py - instrucion tuning of Gemma-7b model
       1.3.2 new_pretrain_gemma.py - pretraining the Gemma-7b model
       1.3.3 gemma_eval.py - evaluating the quality of classification and structured output after full finetuning Gemma-7b model
       1.3.4 pca_base_gemma.py - perplexity score for the base model and model after pretraining

   1.4 Folder 'mistral' - All that was used for two step fine tuning and evaluating Mistral-v0.3-7b model
       1.4.1 instruct_tuning_mistral.py - instrucion tuning of Mistral-v0.3-7b model
       1.4.2 nnew_pretrain_mistral.py - pretraining the Mistral-v0.3-7b model
       1.4.3 mistral_eval.py - evaluating the quality of classification and structured output after full finetuning Mistral-v0.3-7b model
       1.4.4 pca_mistral.py - perplexity score for the base model and model after pretraining

   1.5 Folder 'my_medical_site' - All code needed for building a web-site
       1.5.1 main.py - backend pipeline of the site
       1.5.2 folder 'templates' has file index.html where the frontend code is located
       1.5.3 folder __pycache__ has cach

2. Data.zip - All data used in this project

   2.1 Folder 'few-shot_learning' - data for few-shot learning
   2.2 Folder 'instruction_tuning' - data for instruction tuning
   2.3 Folder 'pretraining' - data for pretraining
       2.3.1 med_data.txt - medical texts
       2.3.2 transcriptions.txt - medical transcriptions

3. Folder 'documentation' - All papers related to this work
   3.1 Operator's_Manual_Diploma_2025.pdf - Operator's_Manual for web-site usage
   3.2 Technical_Requirements_Specification_Diploma.pdf - Technical_Requirements for web-site

       







   
       
   
   
