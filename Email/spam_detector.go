package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
	
	// Rename the first charset import to avoid naming conflict

	"github.com/emersion/go-imap"
	"github.com/emersion/go-imap/client"
	"github.com/emersion/go-message/charset"
	gomail "github.com/emersion/go-message/mail"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/crypto/ssh/terminal"
)

// Config represents the configuration for the email scanner
type Config struct {
	ImapServers      map[string]string `json:"imap_servers"`
	ScanFolders      []string          `json:"scan_folders"`
	MaxEmailsToScan  int               `json:"max_emails_to_scan"`
	SaveResults      bool              `json:"save_results"`
	ResultsDirectory string            `json:"results_directory"`
}

// EmailResult represents the classification result for an email
type EmailResult struct {
	Subject        string `json:"subject"`
	From           string `json:"from"`
	Date           string `json:"date"`
	Classification string `json:"classification"`
}

// RuleFeatures represents the rule-based features extracted from an email
type RuleFeatures struct {
	HasUrgency            bool
	ExcessivePunctuation  bool
	HasSuspiciousPhrase   bool
	HasMoney              bool
	HasAllCaps            bool
}

// Document represents a training document
type Document struct {
	Text  string
	Label string
}

// TFIDFModel represents a simple TF-IDF model
type TFIDFModel struct {
	Vocabulary map[string]int
	IDF        map[string]float64
	Labels     []string
}

// RandomForestModel is a simple representation of a Random Forest classifier
// In a real implementation, you would use a machine learning library
type RandomForestModel struct {
	Features []string
	Labels   []string
}

func main() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("Email Spam Scanner")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("\nThis program scans your email account for potential spam messages.")
	fmt.Println("IMPORTANT: This tool only accesses your emails with your explicit permission.")
	fmt.Println("Your login credentials are used only for the current session and are not stored.")

	// Get consent
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nDo you consent to allowing this program to access your emails for spam detection? (yes/no): ")
	consent, _ := reader.ReadString('\n')
	consent = strings.TrimSpace(strings.ToLower(consent))
	if consent != "yes" && consent != "y" {
		fmt.Println("User consent not provided. Exiting program.")
		return
	}

	// Load configuration
	config := loadConfig()

	// Get credentials
	emailProvider, emailAddress, password := getCredentials()

	// Train a simple model (in a real-world scenario, you'd load a pre-trained model)
	model := trainSimpleModel()

	// Scan emails
	results := scanEmailsForSpam(emailProvider, emailAddress, password, config, model)

	// Display and save results
	if len(results) > 0 {
		displayResults(results)
		saveResults(results, emailAddress, config)
	}
}

// createConfigFile creates a default configuration file
func createConfigFile() {
	config := Config{
		ImapServers: map[string]string{
			"gmail":   "imap.gmail.com",
			"outlook": "outlook.office365.com",
			"yahoo":   "imap.mail.yahoo.com",
		},
		ScanFolders:      []string{"INBOX"},
		MaxEmailsToScan:  50,
		SaveResults:      true,
		ResultsDirectory: "spam_scan_results",
	}

	file, err := json.MarshalIndent(config, "", "    ")
	if err != nil {
		log.Fatalf("Failed to create config file: %v", err)
	}

	err = ioutil.WriteFile("email_scanner_config.json", file, 0644)
	if err != nil {
		log.Fatalf("Failed to write config file: %v", err)
	}

	fmt.Println("Configuration file created: email_scanner_config.json")
	fmt.Println("You can edit this file to customize scanning options.")
}

// loadConfig loads the configuration file or creates a default one
func loadConfig() Config {
	var config Config

	file, err := ioutil.ReadFile("email_scanner_config.json")
	if err != nil {
		fmt.Println("Configuration file not found. Creating default configuration.")
		createConfigFile()
		file, err = ioutil.ReadFile("email_scanner_config.json")
		if err != nil {
			log.Fatalf("Failed to read config file: %v", err)
		}
	}

	err = json.Unmarshal(file, &config)
	if err != nil {
		log.Fatalf("Failed to parse config file: %v", err)
	}

	return config
}

// getCredentials gets email credentials from the user
func getCredentials() (string, string, string) {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\n=== Email Account Access ===")
	fmt.Println("NOTE: Your credentials are used locally only and are not stored.")
	fmt.Println("For Gmail users: You'll need to create an App Password if you have 2FA enabled.")
	fmt.Println("Instructions: https://support.google.com/accounts/answer/185833\n")

	fmt.Print("Email provider (gmail/outlook/yahoo): ")
	emailProvider, _ := reader.ReadString('\n')
	emailProvider = strings.TrimSpace(strings.ToLower(emailProvider))

	fmt.Print("Email address: ")
	emailAddress, _ := reader.ReadString('\n')
	emailAddress = strings.TrimSpace(emailAddress)

	fmt.Print("Password or App Password: ")
	passwordBytes, err := terminal.ReadPassword(int(syscall.Stdin))
	if err != nil {
		log.Fatalf("Failed to read password: %v", err)
	}
	password := string(passwordBytes)
	fmt.Println() // Add a newline after password input

	return emailProvider, emailAddress, password
}

// preprocessText preprocesses the text for the ML model
func preprocessText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove special characters and numbers
	re := regexp.MustCompile(`[^a-zA-Z\s]`)
	text = re.ReplaceAllString(text, "")

	// Simple tokenization by splitting on whitespace
	tokens := strings.Fields(text)

	// Remove common English stopwords (simplified version)
	stopwords := map[string]bool{
		"i": true, "me": true, "my": true, "myself": true, "we": true, "our": true,
		"ours": true, "ourselves": true, "you": true, "your": true, "yours": true,
		"yourself": true, "yourselves": true, "he": true, "him": true, "his": true,
		"himself": true, "she": true, "her": true, "hers": true, "herself": true,
		"it": true, "its": true, "itself": true, "they": true, "them": true,
		"their": true, "theirs": true, "themselves": true, "what": true, "which": true,
		"who": true, "whom": true, "this": true, "that": true, "these": true,
		"those": true, "am": true, "is": true, "are": true, "was": true, "were": true,
		"be": true, "been": true, "being": true, "have": true, "has": true, "had": true,
		"having": true, "do": true, "does": true, "did": true, "doing": true, "a": true,
		"an": true, "the": true, "and": true, "but": true, "if": true, "or": true,
		"because": true, "as": true, "until": true, "while": true, "of": true,
		"at": true, "by": true, "for": true, "with": true, "about": true, "against": true,
		"between": true, "into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "to": true, "from": true,
		"up": true, "down": true, "in": true, "out": true, "on": true, "off": true,
		"over": true, "under": true, "again": true, "further": true, "then": true,
		"once": true,
	}

	filteredTokens := []string{}
	for _, token := range tokens {
		if !stopwords[token] {
			filteredTokens = append(filteredTokens, token)
		}
	}

	return strings.Join(filteredTokens, " ")
}

// trainSimpleModel trains a simple model for spam detection
// In a real implementation, you would use a machine learning library
func trainSimpleModel() *TFIDFModel {
	// Sample training data
	documents := []Document{
		{Text: "Congratulations! You've won a $1000 gift card. Claim now!", Label: "spam"},
		{Text: "Double your income working from home! Limited offer!", Label: "spam"},
	}

	// Preprocess documents
	processedDocs := make([]string, len(documents))
	labels := make([]string, len(documents))
	for i, doc := range documents {
		processedDocs[i] = preprocessText(doc.Text)
		labels[i] = doc.Label
	}

	// Create a simple TF-IDF model
	model := &TFIDFModel{
		Vocabulary: make(map[string]int),
		IDF:        make(map[string]float64),
		Labels:     labels,
	}

	// Build vocabulary
	wordSet := make(map[string]bool)
	for _, doc := range processedDocs {
		words := strings.Fields(doc)
		for _, word := range words {
			wordSet[word] = true
		}
	}

	i := 0
	for word := range wordSet {
		model.Vocabulary[word] = i
		i++
	}

	// This is a very simplified model - in a real implementation,
	// you would compute actual TF-IDF values and train a classifier

	return model
}

// extractRuleFeatures extracts rule-based features from email text
func extractRuleFeatures(text string) RuleFeatures {
	features := RuleFeatures{}

	// Check for urgency keywords
	urgencyWords := []string{"urgent", "immediately", "alert", "warning", "attention", "important"}
	for _, word := range urgencyWords {
		if strings.Contains(strings.ToLower(text), word) {
			features.HasUrgency = true
			break
		}
	}

	// Check for excessive punctuation
	exclamationCount := strings.Count(text, "!") + strings.Count(text, "?")
	features.ExcessivePunctuation = exclamationCount > 2

	// Check for suspicious phrases
	suspiciousPhrases := []string{
		"verify your account", "verify your identity", "confirm your details",
		"click here", "sign up now", "limited time", "act now", "free gift",
		"been selected", "earn money", "winner", "winning", "won",
		"credit card", "password", "bank account", "security", "compromised",
	}
	for _, phrase := range suspiciousPhrases {
		if strings.Contains(strings.ToLower(text), phrase) {
			features.HasSuspiciousPhrase = true
			break
		}
	}

	// Check for money symbols and amounts
	moneyRegex := regexp.MustCompile(`[$€£¥]\d+|\d+[$€£¥]|\$\d+\.\d+`)
	features.HasMoney = moneyRegex.MatchString(text)

	// Check for ALL CAPS words
	allCapsRegex := regexp.MustCompile(`\b[A-Z]{2,}\b`)
	features.HasAllCaps = allCapsRegex.MatchString(text)

	return features
}

// classifyEmail classifies an email as spam or ham
func classifyEmail(emailText string, model *TFIDFModel) string {
	// Preprocess text for ML model
	processedText := preprocessText(emailText)
	_ = processedText
	// This is a placeholder for ML prediction
	// In a real implementation, you would use the model to make a prediction
	mlPrediction := "spam" // Simplified for demonstration

	// Get rule-based features
	rules := extractRuleFeatures(emailText)

	// Combine ML and rule-based approach
	spamIndicators := 0
	if mlPrediction == "spam" {
		spamIndicators++
	}
	if rules.HasUrgency && (rules.HasSuspiciousPhrase || rules.HasMoney) {
		spamIndicators += 2
	}
	if rules.ExcessivePunctuation && rules.HasAllCaps {
		spamIndicators++
	}
	if rules.HasMoney && rules.HasSuspiciousPhrase {
		spamIndicators += 2
	}

	// Special cases for legitimate notifications
	isLikelyLegitimate := false
	legitimatePatterns := []string{
		"meeting scheduled",
		"appointment",
		"will be delivered",
		"package will",
		"your order",
		"invoice",
		"receipt",
		"reservation",
		"reminder:",
	}

	for _, pattern := range legitimatePatterns {
		if strings.Contains(strings.ToLower(emailText), pattern) && !rules.HasMoney && !rules.HasSuspiciousPhrase {
			isLikelyLegitimate = true
			break
		}
	}

	// Final decision
	if isLikelyLegitimate {
		return "Ham Mail (Not Spam)"
	} else if spamIndicators >= 2 {
		return "Spam Mail"
	} else if mlPrediction == "spam" && (rules.HasUrgency || rules.ExcessivePunctuation || rules.HasSuspiciousPhrase || rules.HasMoney || rules.HasAllCaps) {
		return "Spam Mail"
	} else {
		return "Ham Mail (Not Spam)"
	}
}

// extractEmailContent extracts the subject and body from an email message
func extractEmailContent(msg *gomail.Reader) (string, string, error) {
	// Get the header
	header := msg.Header
	subject := header.Get("Subject")
	var body strings.Builder

	// Process each part of the message
	for {
		part, err := msg.NextPart()
		if err != nil {
			break
		}

		switch h := part.Header.(type) {
		case *gomail.Header:
			// Get content type
			contentType, _, _ := h.ContentType()
			
			// Only process text parts
			if strings.HasPrefix(contentType, "text/plain") {
				buf := new(strings.Builder)
				_, err := io.Copy(buf, part.Body)
				if err != nil {
					continue
				}
				body.WriteString(buf.String())
			} else if strings.HasPrefix(contentType, "text/html") && body.Len() == 0 {
				// Use HTML if no plain text is found
				buf := new(strings.Builder)
				_, err := io.Copy(buf, part.Body)
				if err != nil {
					continue
				}
				
				// Simple HTML to text conversion
				htmlBody := buf.String()
				re := regexp.MustCompile(`<[^>]*>`)
				textBody := re.ReplaceAllString(htmlBody, " ")
				body.WriteString(textBody)
			}
		}
	}

	return subject, body.String(), nil
}

// scanEmailsForSpam connects to an email server and scans for spam
func scanEmailsForSpam(emailProvider, emailAddress, password string, config Config, model *TFIDFModel) []EmailResult {
	defaultServers := map[string]string{
		"gmail":   "imap.gmail.com",
		"outlook": "outlook.office365.com",
		"yahoo":   "imap.mail.yahoo.com",
	}

	// Register the UTF-8 charset handler
	charset.RegisterEncoding("utf-8", unicode.UTF8)

	// Get IMAP server
	var imapServer string
	if server, ok := defaultServers[emailProvider]; ok {
		imapServer = server
	} else if server, ok := config.ImapServers[emailProvider]; ok {
		imapServer = server
	} else {
		fmt.Printf("Unknown email provider: %s\n", emailProvider)
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("Please enter IMAP server address: ")
		imapServer, _ = reader.ReadString('\n')
		imapServer = strings.TrimSpace(imapServer)
	}

	fmt.Printf("Connecting to IMAP server: %s\n", imapServer)

	// Connect to the server
	c, err := client.DialTLS(imapServer+":993", nil)
	if err != nil {
		fmt.Printf("Connection error: %v\n", err)
		fmt.Printf("Could not connect to %s. Please check your internet connection.\n", imapServer)
		return nil
	}
	defer c.Logout()

	// Login
	if err := c.Login(emailAddress, password); err != nil {
		fmt.Printf("Login failed: %v\n", err)
		fmt.Println("Please check your credentials and try again.")
		fmt.Println("If using Gmail with 2FA, make sure to use an App Password.")
		return nil
	}

	results := []EmailResult{}

	// Process each folder
	for _, folder := range config.ScanFolders {
		_, err := c.Select(folder, false)
		if err != nil {
			fmt.Printf("Could not access folder: %s\n", folder)
			continue
		}

		// Search for all messages
		criteria := imap.NewSearchCriteria()
		criteria.WithoutFlags = []string{imap.DeletedFlag}
		uids, err := c.Search(criteria)
		if err != nil {
			fmt.Printf("No messages found in %s\n", folder)
			continue
		}

		// Limit the number of emails to scan
		maxEmails := config.MaxEmailsToScan
		if len(uids) > maxEmails {
			uids = uids[len(uids)-maxEmails:]
		}

		fmt.Printf("\nScanning %d emails in %s...\n", len(uids), folder)

		seqSet := new(imap.SeqSet)
		seqSet.AddNum(uids...)

		// Get the whole message body
		section := &imap.BodySectionName{}
		items := []imap.FetchItem{section.FetchItem(), imap.FetchEnvelope}

		messages := make(chan *imap.Message, 10)
		done := make(chan error, 1)
		go func() {
			done <- c.Fetch(seqSet, items, messages)
		}()

		// Process messages
		count := 0
		for msg := range messages {
			count++
			fmt.Printf("\rProcessed %d/%d emails", count, len(uids))

			r := msg.GetBody(section)
			if r == nil {
				continue
			}

			// Parse message
			mr, err := gomail.CreateReader(r)
			if err != nil {
				continue
			}

			// Extract content
			subject, body, err := extractEmailContent(mr)
			if err != nil {
				continue
			}

			// Combine subject and part of body for classification
			classificationText := subject + " " + body
			if len(classificationText) > 500 {
				classificationText = classificationText[:500]
			}

			// Classify the email
			prediction := classifyEmail(classificationText, model)

			// Store results
			from := ""
			date := ""
			if msg.Envelope != nil {
				if msg.Envelope != nil && len(msg.Envelope.From) > 0 {
					addr := msg.Envelope.From[0]
					from = fmt.Sprintf("%s@%s", addr.MailboxName, addr.HostName)
				}
				if msg.Envelope.Date != (time.Time{}) {
					date = msg.Envelope.Date.String()
				}
			}

			result := EmailResult{
				Subject:        subject,
				From:           from,
				Date:           date,
				Classification: prediction,
			}

			results = append(results, result)
		}

		if err := <-done; err != nil {
			fmt.Printf("Error fetching messages: %v\n", err)
		}

		fmt.Printf("\nCompleted scanning folder: %s\n", folder)
	}

	return results
}

// displayResults displays the scan results
func displayResults(results []EmailResult) {
	if len(results) == 0 {
		fmt.Println("No emails were processed.")
		return
	}

	spamCount := 0
	for _, r := range results {
		if r.Classification == "Spam Mail" {
			spamCount++
		}
	}
	hamCount := len(results) - spamCount

	fmt.Println("\n=== Scan Results ===")
	fmt.Printf("Total emails scanned: %d\n", len(results))
	fmt.Printf("Spam emails detected: %d (%.1f%%)\n", spamCount, float64(spamCount)/float64(len(results))*100)
	fmt.Printf("Legitimate emails: %d (%.1f%%)\n", hamCount, float64(hamCount)/float64(len(results))*100)

	if spamCount > 0 {
		fmt.Println("\nPotential spam emails:")
		for i, r := range results {
			if r.Classification == "Spam Mail" {
				fmt.Printf("%d. Subject: %s\n", i+1, r.Subject)
				fmt.Printf("   From: %s\n", r.From)
				fmt.Printf("   Date: %s\n", r.Date)
				fmt.Println(strings.Repeat("-", 50))
			}
		}
	}
}

// saveResults saves the results to a file
func saveResults(results []EmailResult, emailAddress string, config Config) {
	if !config.SaveResults {
		return
	}

	// Create directory if it doesn't exist
	if _, err := os.Stat(config.ResultsDirectory); os.IsNotExist(err) {
		os.MkdirAll(config.ResultsDirectory, 0755)
	}

	// Create filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	emailUser := strings.Split(emailAddress, "@")[0]
	filename := filepath.Join(config.ResultsDirectory, fmt.Sprintf("scan_%s_%s.json", emailUser, timestamp))

	// Convert results to JSON
	jsonData, err := json.MarshalIndent(results, "", "    ")
	if err != nil {
		fmt.Printf("Error saving results: %v\n", err)
		return
	}

	// Write to file
	err = ioutil.WriteFile(filename, jsonData, 0644)
	if err != nil {
		fmt.Printf("Error writing to file: %v\n", err)
		return
	}

	fmt.Printf("\nResults saved to: %s\n", filename)
}
