import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset with labeled emails
emails = [
    # Training data
    {"text": "Congratulations! You've won a $1000 gift card. Claim now!", "label": "spam"},
    {"text": "Double your income working from home! Limited offer!", "label": "spam"},
    {"text": "URGENT: Your account will be suspended. Verify now!", "label": "spam"},
    {"text": "FREE iPhone 15 Pro! You are the 1000th visitor!", "label": "spam"},
    {"text": "Get rich quick! Proven method to earn $10K weekly!", "label": "spam"},
    {"text": "Hot singles in your area! Click to chat now!", "label": "spam"},
    {"text": "URGENT: Security alert! Verify your identity now!", "label": "spam"},
    {"text": "Your package has been shipped, tracking number: AZ8932X", "label": "ham"},
    {"text": "Your monthly statement is now available online", "label": "ham"},
    {"text": "Team meeting scheduled for Friday at 2pm", "label": "ham"},
    {"text": "Your flight confirmation: Departing LAX 10:30 AM", "label": "ham"},
    {"text": "Your prescription is ready for pickup", "label": "ham"},
    {"text": "Payment received, thank you for your purchase", "label": "ham"},
    {"text": "Invoice #12345 for your recent order", "label": "ham"},
    {"text": "Reminder: Doctor's appointment tomorrow at 3pm", "label": "ham"},
    {"text": "Your Amazon order will arrive tomorrow", "label": "ham"},
    # Add more training examples for better coverage
    {"text": "Limited time offer! 80% discount on all products!", "label": "spam"},
    {"text": "URGENT: Your bank account has been locked", "label": "spam"},
    {"text": "You've been selected for a special offer!", "label": "spam"},
    {"text": "Click here to claim your prize now!", "label": "spam"},
    {"text": "Lose 20 pounds in 1 week! New miracle pill!", "label": "spam"},
    {"text": "Your password has expired. Click to update now", "label": "spam"},
    {"text": "Reminder: Project deadline is next Monday", "label": "ham"},
    {"text": "Your subscription has been renewed successfully", "label": "ham"},
    {"text": "Thank you for your payment of $49.99", "label": "ham"},
    {"text": "Your support ticket #45678 has been resolved", "label": "ham"}
]

# Create DataFrame
df = pd.DataFrame(emails)

# Text preprocessing function without NLTK dependencies
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Remove common English stopwords (simplified version)
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once']
    
    tokens = [word for word in tokens if word not in stopwords]
    
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Enhanced spam detection features
def extract_features(text):
    features = {}
    
    # Check for urgency keywords
    urgency_words = ['urgent', 'immediately', 'alert', 'warning', 'attention', 'important']
    features['has_urgency'] = any(word in text.lower() for word in urgency_words)
    
    # Check for excessive punctuation
    features['excessive_punctuation'] = len(re.findall(r'[!?]', text)) > 2
    
    # Check for suspicious phrases
    suspicious_phrases = [
        'verify your account', 'verify your identity', 'confirm your details',
        'click here', 'sign up now', 'limited time', 'act now', 'free gift',
        'been selected', 'earn money', 'winner', 'winning', 'won',
        'credit card', 'password', 'bank account', 'security', 'compromised'
    ]
    features['has_suspicious_phrase'] = any(phrase in text.lower() for phrase in suspicious_phrases)
    
    # Check for money symbols and amounts
    features['has_money'] = bool(re.search(r'[$€£¥]\d+|\d+[$€£¥]|\$\d+\.\d+', text))
    
    # Check for ALL CAPS words
    features['has_all_caps'] = bool(re.search(r'\b[A-Z]{2,}\b', text))
    
    return features

# Function to classify emails with both ML model and rule-based features
def classify_email(email_text):
    # Preprocess text for ML model
    processed_text = preprocess_text(email_text)
    
    # Get ML prediction
    ml_features = vectorizer.transform([processed_text])
    ml_prediction = model.predict(ml_features)[0]
    
    # Get rule-based features
    rules = extract_features(email_text)
    
    # Combine ML and rule-based approach
    # Spam indicators
    spam_indicators = 0
    if ml_prediction == 'spam':
        spam_indicators += 1
    if rules['has_urgency'] and (rules['has_suspicious_phrase'] or rules['has_money']):
        spam_indicators += 2
    if rules['excessive_punctuation'] and rules['has_all_caps']:
        spam_indicators += 1
    if rules['has_money'] and rules['has_suspicious_phrase']:
        spam_indicators += 2
        
    # Special cases for legitimate notifications
    is_likely_legitimate = False
    legitimate_patterns = [
        r'meeting scheduled', 
        r'appointment', 
        r'will be delivered',
        r'package will', 
        r'your order',
        r'invoice',
        r'receipt',
        r'reservation',
        r'reminder:'
    ]
    
    if any(re.search(pattern, email_text.lower()) for pattern in legitimate_patterns) and not rules['has_money'] and not rules['has_suspicious_phrase']:
        is_likely_legitimate = True
    
    # Final decision
    if is_likely_legitimate:
        return "Ham Mail (Not Spam)"
    elif spam_indicators >= 2:
        return "Spam Mail"
    elif ml_prediction == 'spam' and any(rules.values()):
        return "Spam Mail"
    else:
        return "Ham Mail (Not Spam)"

# Test the classifier on the provided examples
test_emails = [
    "Urgent! Your account has been compromised. Verify your identity immediately",
    "Earn $500/day from home! No experience needed. Sign up now",
    "URGENT: Your account has been compromised. Click here to reset your password immediately!",
    "Meeting scheduled for tomorrow at 10 AM. Please confirm your attendance.",
    "FREE GIFT worth $100! Claim your prize now! Limited time offer!!!",
    "Your Amazon package will be delivered tomorrow between 2-4 PM."
]

print("\nClassification Results:")
for i, email in enumerate(test_emails, 1):
    prediction = classify_email(email)
    print(f"Email {i}: {email}")
    print(f"Prediction: {prediction}")
    print("-" * 50)


import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import imaplib
import email
import email.header
import os
import json
from datetime import datetime
import getpass
import sys

# Load or train the spam classifier (using your existing code)
# Note: This section assumes you've already trained the model as in your previous code

# Create a config file to store email settings (not credentials)
def create_config_file():
    config = {
        "imap_servers": {
            "gmail": "imap.gmail.com",
            "outlook": "outlook.office365.com",
            "yahoo": "imap.mail.yahoo.com"
        },
        "scan_folders": ["INBOX"],
        "max_emails_to_scan": 50,
        "save_results": True,
        "results_directory": "spam_scan_results"
    }
    
    with open("email_scanner_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("Configuration file created: email_scanner_config.json")
    print("You can edit this file to customize scanning options.")

# Load configuration
def load_config():
    try:
        with open("email_scanner_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Configuration file not found. Creating default configuration.")
        create_config_file()
        with open("email_scanner_config.json", "r") as f:
            return json.load(f)

# Function to securely get email credentials from user
def get_credentials():
    print("\n=== Email Account Access ===")
    print("NOTE: Your credentials are used locally only and are not stored.")
    print("For Gmail users: You'll need to create an App Password if you have 2FA enabled.")
    print("Instructions: https://support.google.com/accounts/answer/185833\n")
    
    email_provider = input("Email provider (gmail/outlook/yahoo): ").lower()
    email_address = input("Email address: ")
    password = getpass.getpass("Password or App Password: ")
    
    return email_provider, email_address, password

# Function to extract email content
def extract_email_content(msg):
    subject = ""
    body = ""
    
    # Get subject
    subject_header = email.header.make_header(email.header.decode_header(msg.get('Subject', '')))
    subject = str(subject_header)
    
    # Get body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    break
                elif content_type == "text/html" and not body:
                    # Use HTML content if no plain text is found
                    html_body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    # Simple HTML to text conversion (very basic)
                    body = re.sub('<[^<]+?>', ' ', html_body)
    else:
        # Not multipart - get payload directly
        body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
    
    return subject, body

# Connect to email and scan for spam
def scan_emails_for_spam(email_provider, email_address, password, config, classify_email_func):
    default_servers = {
        "gmail": "imap.gmail.com",
        "outlook": "outlook.office365.com",
        "yahoo": "imap.mail.yahoo.com"
    }
    
    # Get IMAP server, using defaults as a fallback
    if email_provider in default_servers:
        imap_server = default_servers[email_provider]
    elif email_provider in config["imap_servers"]:
        imap_server = config["imap_servers"][email_provider]
    else:
        print(f"Unknown email provider: {email_provider}")
        imap_server = input("Please enter IMAP server address: ")
    
    print(f"Connecting to IMAP server: {imap_server}")
    
    # Connect to the server
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_address, password)
    except imaplib.IMAP4.error as e:
        print(f"Login failed: {str(e)}")
        print("Please check your credentials and try again.")
        print("If using Gmail with 2FA, make sure to use an App Password.")
        return
    except Exception as e:
        print(f"Connection error: {str(e)}")
        print(f"Could not connect to {imap_server}. Please check your internet connection.")
        return
    
    results = []
    
    # Process each folder
    for folder in config["scan_folders"]:
        try:
            status, messages = mail.select(folder)
            if status != "OK":
                print(f"Could not access folder: {folder}")
                continue
                
            # Get messages
            status, data = mail.search(None, 'ALL')
            if status != "OK":
                print(f"No messages found in {folder}")
                continue
                
            # Get list of email IDs
            email_ids = data[0].split()
            
            # Limit the number of emails to scan
            email_ids = email_ids[-min(config["max_emails_to_scan"], len(email_ids)):]
            
            print(f"\nScanning {len(email_ids)} emails in {folder}...")
            
            for i, email_id in enumerate(email_ids):
                status, data = mail.fetch(email_id, '(RFC822)')
                
                if status != "OK":
                    continue
                    
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                subject, body = extract_email_content(msg)
                
                # Combine subject and part of body for classification
                classification_text = f"{subject} {body[:500]}"
                
                # Classify the email
                prediction = classify_email_func(classification_text)
                
                # Store results
                from_header = email.header.make_header(email.header.decode_header(msg.get('From', '')))
                date_header = email.header.make_header(email.header.decode_header(msg.get('Date', '')))
                
                result = {
                    "subject": subject,
                    "from": str(from_header),
                    "date": str(date_header),
                    "classification": prediction
                }
                
                results.append(result)
                
                # Display progress
                sys.stdout.write(f"\rProcessed {i+1}/{len(email_ids)} emails")
                sys.stdout.flush()
            
            print("\nCompleted scanning folder:", folder)
            
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
    
    # Close connection
    mail.logout()
    
    return results

# Save results to file
def save_results(results, email_address, config):
    if not config["save_results"]:
        return
        
    # Create directory if it doesn't exist
    if not os.path.exists(config["results_directory"]):
        os.makedirs(config["results_directory"])
        
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config['results_directory']}/scan_{email_address.split('@')[0]}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to: {filename}")

# Display results
def display_results(results):
    if not results:
        print("No emails were processed.")
        return
        
    spam_count = sum(1 for r in results if r["classification"] == "Spam Mail")
    ham_count = sum(1 for r in results if r["classification"] == "Ham Mail (Not Spam)")
    
    print("\n=== Scan Results ===")
    print(f"Total emails scanned: {len(results)}")
    print(f"Spam emails detected: {spam_count} ({spam_count/len(results)*100:.1f}%)")
    print(f"Legitimate emails: {ham_count} ({ham_count/len(results)*100:.1f}%)")
    
    if spam_count > 0:
        print("\nPotential spam emails:")
        for i, r in enumerate(results):
            if r["classification"] == "Spam Mail":
                print(f"{i+1}. Subject: {r['subject']}")
                print(f"   From: {r['from']}")
                print(f"   Date: {r['date']}")
                print("-" * 50)

# Main function
def main():
    print("=" * 60)
    print("Email Spam Scanner")
    print("=" * 60)
    print("\nThis program scans your email account for potential spam messages.")
    print("IMPORTANT: This tool only accesses your emails with your explicit permission.")
    print("Your login credentials are used only for the current session and are not stored.")
    
    # Get consent
    consent = input("\nDo you consent to allowing this program to access your emails for spam detection? (yes/no): ")
    if consent.lower() not in ["yes", "y"]:
        print("User consent not provided. Exiting program.")
        return
    
    # Load configuration
    config = load_config()
    
    # Get credentials
    email_provider, email_address, password = get_credentials()
    
    # Scan emails
    results = scan_emails_for_spam(email_provider, email_address, password, config, classify_email)
    
    # Display and save results
    if results:
        display_results(results)
        save_results(results, email_address, config)

if __name__ == "__main__":
    main()