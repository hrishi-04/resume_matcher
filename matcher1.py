import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import datasets
import transformers
from pypdf import PdfReader
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

punctuation = set(string.punctuation)
stop_words_english = set(stopwords.words("english"))

def reader(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}

    for sent in sentences:
        for criteria in ['skills', 'education', 'job', 'experience','knowledge']:
            if criteria in sent:
                words = word_tokenize(sent)
                words = [word for word in words if word not in stop_words_english]

                # Lemmatize the words
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

                # POS tagger to identify and remove stop words and other irrelevant words
                tagged_words = pos_tag(lemmatized_words)
                filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP', 'CC', 'MD', 'WDT']]
                features['feature'] += " ".join(filtered_words)

    return features

# Example usage
pdf_path = r"data\HR\10694288.pdf"
pdf_text = reader(pdf_path)
preprocessed_text = preprocess_text(pdf_text)
print("Preprocessed text")

"""
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Function to create a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Function to display top words and their frequency
def display_top_words_frequency(text, top_n=10):
    word_frequency = Counter(text.split())
    top_words = dict(word_frequency.most_common(top_n))
    print("Top Words and their Frequency:")
    for word, freq in top_words.items():
        print(f"{word}: {freq}")

# Example usage
preprocessed_text = preprocess_text(pdf_text)['feature']
generate_word_cloud(preprocessed_text)
display_top_words_frequency(preprocessed_text)




resume_info_csv_path = 'Resume.csv'  # Update the path to your CSV file
resume_info_df = pd.read_csv(resume_info_csv_path)

processed_resume_df = pd.DataFrame(columns=['ID', 'Category', 'Processed_Text'])

# Iterate through each row (resume) in the CSV
for index, row in resume_info_df.iterrows():
    id = row['ID']
    category = row['Category']
    
    # Process the PDF file and obtain the processed text
    pdf_path = f"data/{category}/{id}.pdf"  # Update the path to match your directory structure
    pdf_text = reader(pdf_path)
    preprocessed_text = preprocess_text(pdf_text)
    
    # Create a new DataFrame with the current resume information
    current_resume_df = pd.DataFrame({'ID': [id], 'Category': [category], 'Processed_Text': [preprocessed_text['feature']]})
    
    # Concatenate the current resume DataFrame to the processed_resume_df
    processed_resume_df = pd.concat([processed_resume_df, current_resume_df], ignore_index=True)
    

processed_resume_df.to_csv('processed_resumes.csv', index=False)


# Display the processed resume DataFrame
print("Processed Resumes:")
print(processed_resume_df.head())
print("Number of elements in processed_resume_df:", processed_resume_df.shape[0])
print("Number of elements in processed_resume_df:", processed_resume_df.shape[0])



"""

resume_load_df = pd.read_csv('processed_resumes.csv')
print(resume_load_df.head())
print(resume_load_df.shape[0])

from datasets import load_dataset
dataset = load_dataset('jacob-hugging-face/job-descriptions')


# Get the total number of entries in the dataset
total_entries = len(dataset['train'])

# Choose 15 random indices
random_indices = np.random.choice(total_entries, 15, replace=False)

# Convert numpy int32 to int
random_indices = random_indices.astype(int)

random_entries = dataset['train'].select([int(idx) for idx in random_indices])

job_descriptions = []
position_titles = []
company_names = []
for entry in random_entries:
    job_description = entry['job_description']
    position_title = entry['position_title']
    company_name = entry['company_name']
    job_descriptions.append(job_description)
    position_titles.append(position_title)
    company_names.append(company_name)

# Create a DataFrame
data = {'Position_Title': position_titles, 'Job_Description': job_descriptions, 'Company_Name': company_names}
df_final = pd.DataFrame(data)

# Preprocess the descriptions
df_final['Post processing job'] = df_final['Job_Description'].apply(lambda x: preprocess_text(x)['feature'])



print(df_final.shape[0])
print(df_final)


from transformers import AutoModel, AutoTokenizer
import torch

device="cuda"if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#putting model and tokenizer to gpu for faster processing 
model.to(device)

print("reached post model")

"""
def get_embeddings(text):
    inputs = tokenizer(str(text), return_tensors="pt",truncation=True,padding=True).to(device)
    outputs = model(**inputs)
    #getting the embedding to cpu
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().to("cpu").numpy() 
    return embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# Calculate embeddings for all job descriptions and resumes
job_desc_embeddings = np.array([get_embeddings(desc) for desc in df_final['Post processing job']])
resume_embeddings = np.array([get_embeddings(text) for text in resume_load_df['Processed_Text']])

np.save('job_desc_embeddings.npy', job_desc_embeddings)
np.save('resume_embeddings.npy', resume_embeddings)

"""

job_desc_embeddings = np.load('job_desc_embeddings.npy')
resume_embeddings = np.load('resume_embeddings.npy')

#sqeezing job embeding 
job_desc_embeddings=job_desc_embeddings.squeeze()
resume_embeddings=resume_embeddings.squeeze()
resume_embeddings.shape , job_desc_embeddings.shape

# Initialize a DataFrame to store the results
result_df = pd.DataFrame(columns=['jobId', 'resumeId', 'similarity', 'domainResume', 'domainDesc', 'company name'])

# top k-resumes
k=5

from sklearn.metrics.pairwise import cosine_similarity

# Iterate over job descriptions
for i, job_desc_emb in enumerate(job_desc_embeddings):
    job_desc_id = i
    job_title = df_final['Position_Title'].iloc[i]
    company = df_final['Company_Name'].iloc[i]
    # Compute cosine similarities between the current job description and all resumes
    similarities = cosine_similarity([job_desc_emb], resume_embeddings )

    # Get the indices of the top-k most similar resumes
    top_k_indices = np.argsort(similarities[0])[::-1][:k]
   
    # Extract the relevant information and add it to the result DataFrame
    for j in top_k_indices:
        resume_id = resume_load_df['ID'].iloc[j]
        work_domain = resume_load_df['Category'].iloc[j]
        similarity_score = similarities[0][j]
        
        result_df.loc[i+j] = [job_desc_id, resume_id, similarity_score, work_domain,job_title,company]
        

# Sort the results by similarity score (descending)
#result_df = result_df.sort_values(by='similarity', ascending=False)

print(result_df.head())

result_df.to_csv('result.csv', index=False)

df_final.to_csv('dataset.csv', index=False)











