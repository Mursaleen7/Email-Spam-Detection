#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <regex>
#include <iterator>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <cmath>

using namespace std;

struct Features {
    bool has_urgency;
    bool excessive_punctuation;
    bool has_suspicious_phrase;
    bool has_money;
    bool has_all_caps;
};

unordered_set<string> stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", 
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her", 
    "hers", "herself", "it", "it's", "its", "itself", "they", "them", 
    "their", "theirs", "themselves", "what", "which", "who", "whom", 
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", 
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", 
    "because", "as", "until", "while", "of", "at", "by", "for", "with", 
    "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once"
};

Features extract_features(const string& text) {
    Features features;
    string lower_text = text;
    transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

    vector<string> urgency_words = {"urgent", "immediately", "alert", "warning", "attention", "important"};
    features.has_urgency = any_of(urgency_words.begin(), urgency_words.end(), [&lower_text](const string& word) {
        return lower_text.find(word) != string::npos;
    });

    vector<string> suspicious_phrases = {
        "verify your account", "verify your identity", "confirm your details",
        "click here", "sign up now", "limited time", "act now", "free gift",
        "been selected", "earn money", "winner", "winning", "won",
        "credit card", "password", "bank account", "security", "compromised"
    };
    features.has_suspicious_phrase = any_of(suspicious_phrases.begin(), suspicious_phrases.end(), [&lower_text](const string& phrase) {
        return lower_text.find(phrase) != string::npos;
    });

    int punct_count = count_if(lower_text.begin(), lower_text.end(), [](char c) {
        return c == '!' || c == '?';
    });
    features.excessive_punctuation = punct_count > 2;

    regex money_regex(R"(\$?\d+[\.\d+]?[$€£¥]*)");
    smatch match;
    features.has_money = regex_search(lower_text, match, money_regex);

    regex all_caps_regex(R"(\b[A-Z]{2,}\b)");
    features.has_all_caps = regex_search(text, all_caps_regex);

    return features;
}

string preprocess_text(const string& text) {
    string processed;
    for (char c : text) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ' ')
            processed.push_back(tolower(c));
    }
    vector<string> tokens;
    istringstream iss(processed);
    string token;
    while (iss >> token) {
        if (stopwords.find(token) == stopwords.end())
            tokens.push_back(token);
    }
    ostringstream oss;
    for (const string& t : tokens) {
        if (oss.tellp() > 0)
            oss << ' ';
        oss << t;
    }
    return oss.str();
}

struct NaiveBayesModel {
    unordered_map<string, int> spam_word_counts;
    unordered_map<string, int> ham_word_counts;
    int total_spam;
    int total_ham;
    int vocabulary_size;
};

NaiveBayesModel train_naive_bayes(const vector<string>& processed_texts, const vector<string>& labels) {
    unordered_map<string, int> spam_counts;
    unordered_map<string, int> ham_counts;
    int total_spam = 0;
    int total_ham = 0;

    for (size_t i = 0; i < processed_texts.size(); ++i) {
        const string& text = processed_texts[i];
        const string& label = labels[i];
        istringstream iss(text);
        string word;
        while (iss >> word) {
            if (label == "spam") {
                spam_counts[word]++;
                total_spam++;
            } else {
                ham_counts[word]++;
                total_ham++;
            }
        }
    }

    unordered_set<string> vocabulary;
    for (const auto& pair : spam_counts) {
        vocabulary.insert(pair.first);
    }
    for (const auto& pair : ham_counts) {
        vocabulary.insert(pair.first);
    }
    int V = vocabulary.size();

    return {spam_counts, ham_counts, total_spam, total_ham, V};
}

double calculate_log_probability(const string& word, const unordered_map<string, int>& word_counts, int total, int V) {
    auto it = word_counts.find(word);
    int count = (it != word_counts.end()) ? it->second : 0;
    return log((count + 1.0) / (total + V));
}

string classify_naive_bayes(const NaiveBayesModel& model, const string& processed_email) {
    double log_spam = log(static_cast<double>(model.total_spam) / (model.total_spam + model.total_ham));
    double log_ham = log(static_cast<double>(model.total_ham) / (model.total_spam + model.total_ham));

    istringstream iss(processed_email);
    string word;
    while (iss >> word) {
        log_spam += calculate_log_probability(word, model.spam_word_counts, model.total_spam, model.vocabulary_size);
        log_ham += calculate_log_probability(word, model.ham_word_counts, model.total_ham, model.vocabulary_size);
    }

    return log_spam > log_ham ? "Spam Mail" : "Ham Mail (Not Spam)";
}

int main() {
    // Read spam.csv file
    vector<string> labels;
    vector<string> texts;
    ifstream file("spam.csv");
    if (!file) {
        cerr << "Error opening spam.csv" << endl;
        return 1;
    }
    string line;
    getline(file, line); // Skip header
    while (getline(file, line)) {
        size_t comma = line.find(',');
        string label = line.substr(0, comma);
        string text = line.substr(comma + 1);
        labels.push_back(label);
        texts.push_back(text);
    }

    // Preprocess the texts
    vector<string> processed_texts;
    for (const string& text : texts) {
        processed_texts.push_back(preprocess_text(text));
    }

    // Train Naive Bayes model
    NaiveBayesModel model = train_naive_bayes(processed_texts, labels);

    // Test the classifier on the provided examples
    vector<string> test_emails = {
        "Urgent! Your account has been compromised. Verify your identity immediately",
        "Earn $500/day from home! No experience needed. Sign up now",
        "URGENT: Your account has been compromised. Click here to reset your password immediately!",
        "Meeting scheduled for tomorrow at 10 AM. Please confirm your attendance.",
        "FREE GIFT worth $100! Claim your prize now! Limited time offer!!!",
        "Your Amazon package will be delivered tomorrow between 2-4 PM."
    };

    cout << "\nClassification Results:\n" << endl;
    for (size_t i = 0; i < test_emails.size(); ++i) {
        cout << "Email " << (i+1) << ": " << test_emails[i] << "\n";
        string processed_email = preprocess_text(test_emails[i]);
        string prediction = classify_naive_bayes(model, processed_email);
        cout << "Prediction: " << prediction << "\n";
        cout << string(50, '-') << endl;
    }

    return 0;
}