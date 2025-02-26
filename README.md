# Email Spam Scanner

The Email Spam Scanner is a Python-based tool that combines machine learning and rule-based heuristics to detect potential spam emails in your inbox. It connects to your email account using the IMAP protocol, processes your emails, and classifies them as either "Spam Mail" or "Ham Mail (Not Spam)".

## Features

- **Hybrid Spam Detection:** Combines a trained RandomForestClassifier (using TF-IDF features) with custom rule-based checks (e.g., urgency keywords, suspicious phrases, excessive punctuation) for robust spam detection.
- **Email Scanning:** Connects securely to your email provider (Gmail, Outlook, Yahoo, or custom IMAP servers) and scans specified folders (e.g., INBOX).
- **Customizable Configuration:** Includes a configuration file (`email_scanner_config.json`) that allows you to customize scanning options such as folders to scan, the maximum number of emails, and whether to save the results.
- **Result Reporting:** Displays a summary of the scan results in the terminal and saves detailed results as a timestamped JSON file.
- **Privacy First:** Your email credentials are used only for the duration of the session and are not stored.

## Prerequisites

- Python 3.6 or higher
- The following Python packages:
  - `pandas`
  - `scikit-learn`
  - `imaplib` (standard library)
  - `email` (standard library)
  - `getpass` (standard library)
  - `json` (standard library)
  - `re` (standard library)
  - `sys` (standard library)
  - `os` (standard library)
  - `datetime` (standard library)

You can install the necessary third-party packages using pip:

```bash
pip install pandas scikit-learn
