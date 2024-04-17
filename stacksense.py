import os
import requests
import json
import re
from bs4 import BeautifulSoup

# Your StackExchange API key
API_KEY = os.environ['STACKEXCHANGE_API_KEY']
print(API_KEY)

# Function to fetch StackOverflow questions using the API
def fetch_stackoverflow_questions(pages=3):  # Fetching 3 pages (90 questions)
    questions = []
    
    for page in range(1, pages + 1):
        url = f"https://api.stackexchange.com/2.3/questions?page={page}&pagesize=30&order=desc&sort=activity&site=stackoverflow"
        
        headers = {
            "User-Agent": "StackSense",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            questions.extend(data.get('items', []))
        else:
            print(f"Failed to fetch data for page {page}: {response.status_code}")
    
    return questions

# Function to fetch full question body using question ID
def fetch_question_body(question_id):
    url = f"https://api.stackexchange.com/2.3/questions/{question_id}?site=stackoverflow&filter=withbody"
    
    headers = {
        "User-Agent": "StackOverflow Scraper",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data['items'][0]['body'] if data['items'] else None
    else:
        print(f"Failed to fetch question body: {response.status_code}")
        return None

# Function to clean text (remove HTML tags, special characters, etc.)
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

if __name__ == "__main__":
    pages = 3  # Number of pages to fetch (each page has 30 questions)
    
    questions = fetch_stackoverflow_questions(pages)
    
    data_to_write = []
    
    for idx, question in enumerate(questions, 1):
        title = question.get('title', '')
        cleaned_title = clean_text(title)
        
        tags = question.get('tags', [])
        
        question_id = question.get('question_id', '')
        body = fetch_question_body(question_id)
        cleaned_body = clean_text(body) if body else "N/A"
        
        # Prepare data in desired format
        question_data = {
            "title": cleaned_title,
            "tags": tags,
            "body": cleaned_body
        }
        
        data_to_write.append(question_data)
    
    # Write data to JSON file
    with open("stackoverflow_questions.json", "w") as json_file:
        json.dump(data_to_write, json_file, indent=4)
    
    print("Data written to 'stackoverflow_questions.json'")

