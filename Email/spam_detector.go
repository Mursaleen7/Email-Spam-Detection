package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/emersion/go-imap"
	"github.com/emersion/go-imap/client"
	"github.com/emersion/go-message/charset"
	gomail "github.com/emersion/go-message/mail"
	"github.com/fatih/color"
	"github.com/joho/godotenv"
	"golang.org/x/crypto/ssh/terminal"
	"golang.org/x/text/encoding/unicode"
)

// VERSION is the software version
const VERSION = "2.0.0"

// Config represents the enhanced configuration for the email scanner
type Config struct {
	ImapServers        map[string]string `json:"imap_servers"`
	ScanFolders        []string          `json:"scan_folders"`
	MaxEmailsToScan    int               `json:"max_emails_to_scan"`
	SaveResults        bool              `json:"save_results"`
	ResultsDirectory   string            `json:"results_directory"`
	ModelPath          string            `json:"model_path"`
	SpamThreshold      float64           `json:"spam_threshold"`
	UseCache           bool              `json:"use_cache"`
	CachePath          string            `json:"cache_path"`
	ConcurrentWorkers  int               `json:"concurrent_workers"`
	VerboseLogging     bool              `json:"verbose_logging"`
	PhishingDomains    []string          `json:"phishing_domains"`
	AutoLearn          bool              `json:"auto_learn"`
	ScanAttachments    bool              `json:"scan_attachments"`
	EnableHeaderCheck  bool              `json:"enable_header_check"`
	AnalyzeImageLinks  bool              `json:"analyze_image_links"`
	EnableReporting    bool              `json:"enable_reporting"`
	WhitelistedDomains []string          `json:"whitelisted_domains"`
	BlacklistedWords   []string          `json:"blacklisted_words"`
}

// EmailResult represents the enhanced classification result for an email
type EmailResult struct {
	Subject           string             `json:"subject"`
	From              string             `json:"from"`
	Date              string             `json:"date"`
	Classification    string             `json:"classification"`
	SpamScore         float64            `json:"spam_score"`
	Confidence        float64            `json:"confidence"`
	Features          map[string]float64 `json:"features"`
	Rules             []string           `json:"triggered_rules"`
	HTMLPresent       bool               `json:"html_present"`
	NumLinks          int                `json:"num_links"`
	NumImages         int                `json:"num_images"`
	HasAttachments    bool               `json:"has_attachments"`
	HeaderAnalysis    HeaderAnalysis     `json:"header_analysis"`
	ProcessingTime    float64            `json:"processing_time_ms"`
	RawSize           int                `json:"raw_size_bytes"`
	ActionTaken       string             `json:"action_taken,omitempty"`
	ClassifierResults map[string]float64 `json:"classifier_results"`
}

// HeaderAnalysis contains DKIM, SPF and DMARC results
type HeaderAnalysis struct {
	HasSPF    bool   `json:"has_spf"`
	SPFResult string `json:"spf_result,omitempty"`
	HasDKIM   bool   `json:"has_dkim"`
	DKIMValid bool   `json:"dkim_valid,omitempty"`
	HasDMARC  bool   `json:"has_dmarc"`
	DMARCPass bool   `json:"dmarc_pass,omitempty"`
}

// SimpleTokenizer is a basic word tokenizer to replace the prose dependency
type SimpleTokenizer struct{}

// Tokenize splits text into words
func (t *SimpleTokenizer) Tokenize(text string) []string {
	// Simple word boundary tokenization
	re := regexp.MustCompile(`\b\w+\b`)
	return re.FindAllString(text, -1)
}

// FeatureExtractor extracts features from email content
type FeatureExtractor struct {
	tokenizer          *SimpleTokenizer
	urlRegex           *regexp.Regexp
	moneyRegex         *regexp.Regexp
	excessivePunctRegx *regexp.Regexp
	allCapsRegex       *regexp.Regexp
	ipUrlRegex         *regexp.Regexp
	emailRegex         *regexp.Regexp
	stopwords          map[string]bool
	suspiciousPhrases  []string
	urgencyPhrases     []string
	blacklistedWords   []string
	phishingDomains    []string
	whitelistedDomains []string
}

// EnhancedModel represents a more sophisticated model for spam detection
type EnhancedModel struct {
	Vocabulary      map[string]int                `json:"vocabulary"`
	FeatureWeights  map[string]float64            `json:"feature_weights"`
	SpamThreshold   float64                       `json:"spam_threshold"`
	ClassPriors     map[string]float64            `json:"class_priors"`
	WordFrequencies map[string]map[string]int     `json:"word_frequencies"`
	WordProbs       map[string]map[string]float64 `json:"word_probs"`
	NGramProbs      map[string]map[string]float64 `json:"ngram_probs"`
	Rules           []Rule                        `json:"rules"`
	Version         string                        `json:"version"`
	TrainingSize    int                           `json:"training_size"`
	FeatureCount    int                           `json:"feature_count"`
	LastUpdated     time.Time                     `json:"last_updated"`
}

// Rule represents a spam detection rule
type Rule struct {
	Name        string         `json:"name"`
	Pattern     string         `json:"pattern"`
	Regex       *regexp.Regexp `json:"-"`
	Weight      float64        `json:"weight"`
	Description string         `json:"description"`
	IsActive    bool           `json:"is_active"`
}

// Document represents a training document with enhanced features
type Document struct {
	Text     string             `json:"text"`
	Label    string             `json:"label"`
	Features map[string]float64 `json:"features"`
}

// Cache represents a local cache for processed emails
type Cache struct {
	EmailHashes map[string]EmailResult `json:"email_hashes"`
	LastUpdate  time.Time              `json:"last_update"`
	Size        int                    `json:"size"`
}

// Statistics tracks processing statistics
type Statistics struct {
	TotalEmails       int
	SpamEmails        int
	HamEmails         int
	ProcessingTimeMs  float64
	AverageConfidence float64
	StartTime         time.Time
	mutex             sync.Mutex
}

// Add thread-safe statistics update
func (s *Statistics) Add(isSpam bool, processingTime float64, confidence float64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalEmails++
	if isSpam {
		s.SpamEmails++
	} else {
		s.HamEmails++
	}
	s.ProcessingTimeMs += processingTime
	s.AverageConfidence += confidence
}

// PrintSummary prints the statistics summary
func (s *Statistics) PrintSummary() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	duration := time.Since(s.StartTime)
	avgTime := 0.0
	avgConfidence := 0.0

	if s.TotalEmails > 0 {
		avgTime = s.ProcessingTimeMs / float64(s.TotalEmails)
		avgConfidence = s.AverageConfidence / float64(s.TotalEmails)
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("SCAN SUMMARY")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total emails scanned: %d\n", s.TotalEmails)
	spam := color.New(color.FgRed).SprintFunc()
	ham := color.New(color.FgGreen).SprintFunc()
	fmt.Printf("Spam emails detected: %s (%d - %.1f%%)\n",
		spam(s.SpamEmails), s.SpamEmails,
		getPercentage(s.SpamEmails, s.TotalEmails))
	fmt.Printf("Legitimate emails: %s (%d - %.1f%%)\n",
		ham(s.HamEmails), s.HamEmails,
		getPercentage(s.HamEmails, s.TotalEmails))
	fmt.Printf("Total processing time: %.2f seconds\n", duration.Seconds())
	fmt.Printf("Average processing time per email: %.2f ms\n", avgTime)
	fmt.Printf("Average confidence: %.2f%%\n", avgConfidence*100)
}

// Global variables
var (
	stats     Statistics
	cache     Cache
	extractor FeatureExtractor
	logger    *log.Logger
)

func main() {
	// Initialize colorized output
	color.New(color.FgCyan).Add(color.Bold).Println(strings.Repeat("=", 60))
	color.New(color.FgCyan).Add(color.Bold).Println("Enhanced Email Spam Scanner v" + VERSION)
	color.New(color.FgCyan).Add(color.Bold).Println(strings.Repeat("=", 60))
	fmt.Println("\nThis advanced program scans your email account for potential spam messages.")
	fmt.Println("It uses machine learning, header analysis, and sophisticated rule patterns")
	fmt.Println("to identify spam with higher accuracy.")

	disclaimer := color.New(color.FgYellow).Add(color.Bold)
	disclaimer.Println("\nIMPORTANT: This tool only accesses your emails with your explicit permission.")
	disclaimer.Println("Your login credentials are used only for the current session and are not stored.")

	// Initialize logger
	logFile, err := os.OpenFile("spam_scanner.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err == nil {
		logger = log.New(logFile, "SPAM-SCANNER: ", log.Ldate|log.Ltime|log.Lshortfile)
	} else {
		logger = log.New(os.Stderr, "SPAM-SCANNER: ", log.Ldate|log.Ltime|log.Lshortfile)
	}

	// Load .env file if it exists
	_ = godotenv.Load()

	// Initialize statistics
	stats = Statistics{StartTime: time.Now()}

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

	// Initialize feature extractor
	initFeatureExtractor(&config)

	// Initialize cache if enabled
	if config.UseCache {
		loadCache(config.CachePath)
	}

	// Get credentials
	emailProvider, emailAddress, password := getCredentials()

	// Train enhanced model
	model := trainEnhancedModel("spam.csv", config.ModelPath, config.AutoLearn)

	// Scan emails with enhanced scanner
	results := scanEmailsForSpam(emailProvider, emailAddress, password, config, model)

	// Display and save enhanced results
	if len(results) > 0 {
		displayEnhancedResults(results, config)
		saveEnhancedResults(results, emailAddress, config)
	}

	// Save cache if enabled
	if config.UseCache {
		saveCache(config.CachePath)
	}

	// Print final statistics
	stats.PrintSummary()
}

// initFeatureExtractor initializes the feature extractor
func initFeatureExtractor(config *Config) {
	extractor = FeatureExtractor{
		tokenizer:          &SimpleTokenizer{},
		urlRegex:           regexp.MustCompile(`https?://[^\s]+`),
		moneyRegex:         regexp.MustCompile(`[$€£¥]\d+(?:\.\d+)?|\d+(?:\.\d+)?[$€£¥]`),
		excessivePunctRegx: regexp.MustCompile(`[!?]{2,}`),
		allCapsRegex:       regexp.MustCompile(`\b[A-Z]{3,}\b`),
		ipUrlRegex:         regexp.MustCompile(`https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`),
		emailRegex:         regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`),
		stopwords:          loadStopwords(),
		suspiciousPhrases:  loadSuspiciousPhrases(),
		urgencyPhrases:     loadUrgencyPhrases(),
		blacklistedWords:   config.BlacklistedWords,
		phishingDomains:    config.PhishingDomains,
		whitelistedDomains: config.WhitelistedDomains,
	}
}

// loadStopwords loads English stopwords
func loadStopwords() map[string]bool {
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
		"once": true, "here": true, "there": true, "when": true, "where": true, "why": true,
		"how": true, "all": true, "any": true, "both": true, "each": true, "few": true,
		"more": true, "most": true, "other": true, "some": true, "such": true, "no": true,
		"nor": true, "not": true, "only": true, "own": true, "same": true, "so": true,
		"than": true, "too": true, "very": true, "s": true, "t": true, "can": true,
		"will": true, "just": true, "don": true, "should": true, "now": true,
	}
	return stopwords
}

// loadSuspiciousPhrases loads phrases indicative of spam
func loadSuspiciousPhrases() []string {
	return []string{
		"verify your account", "verify your identity", "confirm your details",
		"click here", "sign up now", "limited time", "act now", "free gift",
		"been selected", "earn money", "winner", "winning", "won", "congratulations",
		"credit card", "password reset", "bank account", "security alert", "compromised",
		"suspicious activity", "verify your payment", "verify your billing", "invoice attached",
		"payment confirmation", "update your information", "unusual login", "prize claim",
		"lottery winner", "unclaimed inheritance", "million dollar", "million euros",
		"business proposal", "investment opportunity", "work from home", "make money fast",
		"double your income", "best rates", "risk free", "satisfaction guaranteed",
		"no risk", "100% free", "all natural", "amazing", "cancel at any time",
		"cash bonus", "cheap", "claims not to be spam", "direct email", "direct marketing",
		"fantastic deal", "for only", "free consultation", "free hosting", "free info",
		"free investment", "great offer", "increase sales", "increase traffic",
		"incredible deal", "lowest price", "luxury", "marketing solution",
		"mass email", "meet singles", "multi level marketing", "no catch",
		"no fees", "no obligation", "no purchase necessary", "no questions asked",
		"no strings attached", "not spam", "obligation free", "once in lifetime",
		"order now", "pre-approved", "presently", "promise", "pure profit",
		"risk-free", "satisfaction", "special promotion", "supplies", "take action now",
		"terms and conditions", "the best rates", "trial", "unlimited", "warranty",
		"web traffic", "weekend getaway", "while supplies last", "while stocks last",
		"you are a winner", "you have been selected",
	}
}

// loadUrgencyPhrases loads phrases indicating urgency
func loadUrgencyPhrases() []string {
	return []string{
		"urgent", "immediately", "alert", "warning", "attention", "important",
		"critical", "act now", "expires today", "expires soon", "limited time",
		"don't delay", "last chance", "final notice", "time sensitive",
		"deadline", "urgent response needed", "respond now", "action required",
		"limited offer", "expires in", "only today", "few hours left",
		"don't miss out", "don't wait", "hurry", "instant", "instant access",
		"instant approval", "now only", "now", "offer expires", "once in a lifetime",
		"order now", "special promotion", "supplies are limited", "take action",
		"time limited", "today", "today only", "urgent response", "while stocks last",
	}
}

// createConfigFile creates an enhanced default configuration file
func createConfigFile() {
	config := Config{
		ImapServers: map[string]string{
			"gmail":   "imap.gmail.com",
			"outlook": "outlook.office365.com",
			"yahoo":   "imap.mail.yahoo.com",
			"aol":     "imap.aol.com",
			"zoho":    "imap.zoho.com",
			"proton":  "imap.protonmail.ch",
		},
		ScanFolders:       []string{"INBOX", "Junk", "Spam"},
		MaxEmailsToScan:   100,
		SaveResults:       true,
		ResultsDirectory:  "spam_scan_results",
		ModelPath:         "spam_model.json",
		SpamThreshold:     0.75,
		UseCache:          true,
		CachePath:         "email_cache.json",
		ConcurrentWorkers: 5,
		VerboseLogging:    false,
		PhishingDomains: []string{
			"secure-login", "account-verify", "signin-secure", "banking-alert",
		},
		AutoLearn:         true,
		ScanAttachments:   true,
		EnableHeaderCheck: true,
		AnalyzeImageLinks: true,
		EnableReporting:   true,
		WhitelistedDomains: []string{
			"google.com", "microsoft.com", "apple.com", "amazon.com", "github.com",
			"linkedin.com", "twitter.com", "facebook.com", "instagram.com",
		},
		BlacklistedWords: []string{
			"viagra", "cialis", "enlargement", "pharmaceutical", "pharmacy", "prescription",
			"meds", "medication", "drugs", "adult", "porn", "xxx", "sex", "sexy", "hot singles",
			"dating", "cryptocurrency", "bitcoin", "forex", "trading", "casino", "gambling",
			"lottery", "diet", "weight loss", "slim", "miracle cure", "miracle pill",
		},
	}

	file, err := json.MarshalIndent(config, "", "    ")
	if err != nil {
		logger.Printf("Failed to create config file: %v", err)
		return
	}

	err = ioutil.WriteFile("email_scanner_config.json", file, 0644)
	if err != nil {
		logger.Printf("Failed to write config file: %v", err)
		return
	}

	fmt.Println("Enhanced configuration file created: email_scanner_config.json")
	fmt.Println("You can edit this file to customize scanning options.")
}

// loadConfig loads the enhanced configuration file or creates a default one
func loadConfig() Config {
	var config Config

	file, err := ioutil.ReadFile("email_scanner_config.json")
	if err != nil {
		fmt.Println("Configuration file not found. Creating enhanced default configuration.")
		createConfigFile()
		file, err = ioutil.ReadFile("email_scanner_config.json")
		if err != nil {
			logger.Fatalf("Failed to read config file: %v", err)
		}
	}

	err = json.Unmarshal(file, &config)
	if err != nil {
		logger.Fatalf("Failed to parse config file: %v", err)
	}

	// Set sensible defaults if not provided
	if config.ConcurrentWorkers <= 0 {
		config.ConcurrentWorkers = 5
	}
	if config.SpamThreshold <= 0 {
		config.SpamThreshold = 0.75
	}

	return config
}

// getCredentials gets email credentials from the user with enhanced security
func getCredentials() (string, string, string) {
	reader := bufio.NewReader(os.Stdin)

	headerStyle := color.New(color.FgCyan).Add(color.Bold)
	headerStyle.Println("\n=== Email Account Access ===")
	fmt.Println("NOTE: Your credentials are used locally only and are not stored.")
	fmt.Println("For Gmail users: You'll need to create an App Password if you have 2FA enabled.")
	fmt.Println("Instructions: https://support.google.com/accounts/answer/185833\n")

	// Check for environment variables first
	emailProvider := os.Getenv("EMAIL_PROVIDER")
	emailAddress := os.Getenv("EMAIL_ADDRESS")
	password := os.Getenv("EMAIL_PASSWORD")

	// If env vars are not set, prompt the user
	if emailProvider == "" {
		fmt.Print("Email provider (gmail/outlook/yahoo/aol/zoho/proton): ")
		emailProvider, _ = reader.ReadString('\n')
		emailProvider = strings.TrimSpace(strings.ToLower(emailProvider))
	} else {
		fmt.Printf("Email provider: %s (from environment)\n", emailProvider)
	}

	if emailAddress == "" {
		fmt.Print("Email address: ")
		emailAddress, _ = reader.ReadString('\n')
		emailAddress = strings.TrimSpace(emailAddress)
	} else {
		fmt.Printf("Email address: %s (from environment)\n", emailAddress)
	}

	if password == "" {
		fmt.Print("Password or App Password: ")
		passwordBytes, err := terminal.ReadPassword(int(syscall.Stdin))
		if err != nil {
			logger.Fatalf("Failed to read password: %v", err)
		}
		password = string(passwordBytes)
		fmt.Println() // Add a newline after password input
	} else {
		fmt.Println("Password: [LOADED FROM ENVIRONMENT]")
	}

	return emailProvider, emailAddress, password
}

// loadCache loads the email cache
func loadCache(cachePath string) {
	cache = Cache{
		EmailHashes: make(map[string]EmailResult),
		LastUpdate:  time.Now(),
		Size:        0,
	}

	cacheFile, err := ioutil.ReadFile(cachePath)
	if err != nil {
		logger.Printf("No cache file found, creating new cache")
		return
	}

	err = json.Unmarshal(cacheFile, &cache)
	if err != nil {
		logger.Printf("Error parsing cache file, creating new cache: %v", err)
		return
	}

	logger.Printf("Loaded cache with %d entries", len(cache.EmailHashes))
}

// saveCache saves the email cache
func saveCache(cachePath string) {
	cacheFile, err := json.MarshalIndent(cache, "", "    ")
	if err != nil {
		logger.Printf("Failed to create cache file: %v", err)
		return
	}

	err = ioutil.WriteFile(cachePath, cacheFile, 0644)
	if err != nil {
		logger.Printf("Failed to write cache file: %v", err)
		return
	}

	logger.Printf("Saved cache with %d entries", len(cache.EmailHashes))
}

// generateEmailHash generates a unique hash for an email
func generateEmailHash(subject, from, date string) string {
	return fmt.Sprintf("%x", len(subject)+len(from)+len(date))
}

// extractNGrams extracts n-grams from text
func extractNGrams(text string, n int) []string {
	words := strings.Fields(text)
	if len(words) < n {
		return []string{}
	}

	ngrams := make([]string, 0, len(words)-n+1)
	for i := 0; i <= len(words)-n; i++ {
		ngram := strings.Join(words[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}

	return ngrams
}

// preprocessText preprocesses the text for the enhanced ML model
func preprocessText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Replace URLs and emails with placeholders to reduce noise
	text = extractor.urlRegex.ReplaceAllString(text, "URL_PLACEHOLDER")
	text = extractor.emailRegex.ReplaceAllString(text, "EMAIL_PLACEHOLDER")

	// Remove HTML tags if present
	htmlTagRegex := regexp.MustCompile(`<[^>]*>`)
	text = htmlTagRegex.ReplaceAllString(text, " ")

	// Decode HTML entities
	text = html.UnescapeString(text)

	// Remove special characters and numbers
	re := regexp.MustCompile(`[^a-zA-Z\s]`)
	text = re.ReplaceAllString(text, " ")

	// Tokenize
	tokens := extractor.tokenizer.Tokenize(text)

	// Remove stopwords and single-character words
	filteredTokens := make([]string, 0, len(tokens))
	for _, token := range tokens {
		token = strings.TrimSpace(token)
		if len(token) > 1 && !extractor.stopwords[token] {
			filteredTokens = append(filteredTokens, token)
		}
	}

	return strings.Join(filteredTokens, " ")
}

// extractFeatures extracts enhanced features from email text and metadata
func extractFeatures(subject, body, from, headers string) map[string]float64 {
	features := make(map[string]float64)

	// Combine subject and body for text analysis
	fullText := subject + " " + body

	// Basic text features
	features["text_length"] = float64(len(fullText))
	features["subject_length"] = float64(len(subject))

	// URL features
	urlMatches := extractor.urlRegex.FindAllString(fullText, -1)
	features["url_count"] = float64(len(urlMatches))

	// Check for IP-based URLs (suspicious)
	ipUrlMatches := extractor.ipUrlRegex.FindAllString(fullText, -1)
	features["ip_url_count"] = float64(len(ipUrlMatches))

	// Check URLs against phishing domains
	phishingUrlCount := 0
	for _, urlMatch := range urlMatches {
		for _, phishingDomain := range extractor.phishingDomains {
			if strings.Contains(urlMatch, phishingDomain) {
				phishingUrlCount++
				break
			}
		}
	}
	features["phishing_url_count"] = float64(phishingUrlCount)

	// Money references
	moneyMatches := extractor.moneyRegex.FindAllString(fullText, -1)
	features["money_references"] = float64(len(moneyMatches))

	// Urgency indicators
	urgencyCount := 0
	for _, phrase := range extractor.urgencyPhrases {
		if strings.Contains(strings.ToLower(fullText), phrase) {
			urgencyCount++
		}
	}
	features["urgency_count"] = float64(urgencyCount)

	// Suspicious phrases
	suspiciousPhraseCount := 0
	for _, phrase := range extractor.suspiciousPhrases {
		if strings.Contains(strings.ToLower(fullText), phrase) {
			suspiciousPhraseCount++
		}
	}
	features["suspicious_phrase_count"] = float64(suspiciousPhraseCount)

	// Blacklisted words
	blacklistedWordCount := 0
	for _, word := range extractor.blacklistedWords {
		wordRegex := regexp.MustCompile(`\b` + regexp.QuoteMeta(word) + `\b`)
		matches := wordRegex.FindAllString(strings.ToLower(fullText), -1)
		blacklistedWordCount += len(matches)
	}
	features["blacklisted_word_count"] = float64(blacklistedWordCount)

	// Check for excessive punctuation
	excessivePunctMatches := extractor.excessivePunctRegx.FindAllString(fullText, -1)
	features["excessive_punctuation"] = float64(len(excessivePunctMatches))

	// Check for ALL CAPS (shouting)
	allCapsMatches := extractor.allCapsRegex.FindAllString(fullText, -1)
	features["all_caps_count"] = float64(len(allCapsMatches))

	// Check from domain against whitelisted domains
	fromDomain := extractDomainFromEmail(from)
	isWhitelisted := 0.0
	for _, domain := range extractor.whitelistedDomains {
		if strings.Contains(fromDomain, domain) {
			isWhitelisted = 1.0
			break
		}
	}
	features["is_whitelisted"] = isWhitelisted

	// Check header features
	features["has_spf"] = boolToFloat(strings.Contains(headers, "SPF"))
	features["has_dkim"] = boolToFloat(strings.Contains(headers, "DKIM"))
	features["has_dmarc"] = boolToFloat(strings.Contains(headers, "DMARC"))

	// Subject has RE: or FWD: (less likely to be spam)
	features["is_reply_or_forward"] = boolToFloat(
		strings.HasPrefix(strings.ToLower(subject), "re:") ||
			strings.HasPrefix(strings.ToLower(subject), "fwd:") ||
			strings.HasPrefix(strings.ToLower(subject), "fw:"))

	// Count unique words as a ratio to total words (spam often repeats words)
	words := strings.Fields(strings.ToLower(fullText))
	uniqueWords := make(map[string]bool)
	for _, word := range words {
		uniqueWords[word] = true
	}

	if len(words) > 0 {
		features["unique_word_ratio"] = float64(len(uniqueWords)) / float64(len(words))
	} else {
		features["unique_word_ratio"] = 0
	}

	// HTML features
	features["has_html"] = boolToFloat(strings.Contains(fullText, "<html") ||
		strings.Contains(fullText, "<body") ||
		strings.Contains(fullText, "<div"))

	// Simple HTML link count
	htmlLinkRegex := regexp.MustCompile(`<a\s+[^>]*href=`)
	htmlLinks := htmlLinkRegex.FindAllString(fullText, -1)
	features["html_link_count"] = float64(len(htmlLinks))

	// Simple image count
	imgRegex := regexp.MustCompile(`<img\s+[^>]*src=`)
	imgTags := imgRegex.FindAllString(fullText, -1)
	features["image_count"] = float64(len(imgTags))

	return features
}

// boolToFloat converts a boolean to a float (1.0 for true, 0.0 for false)
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// extractDomainFromEmail extracts the domain part from an email address
func extractDomainFromEmail(email string) string {
	parts := strings.Split(email, "@")
	if len(parts) > 1 {
		return parts[1]
	}
	return ""
}

// trainEnhancedModel trains an enhanced machine learning model
func trainEnhancedModel(filePath string, modelPath string, autoLearn bool) *EnhancedModel {
	// Try to load existing model if it exists
	model := loadModelIfExists(modelPath)
	if model != nil {
		fmt.Printf("Loaded existing model with %d features and %d training examples\n",
			model.FeatureCount, model.TrainingSize)
		return model
	}

	// Create a new model
	model = &EnhancedModel{
		Vocabulary:      make(map[string]int),
		FeatureWeights:  make(map[string]float64),
		SpamThreshold:   0.75,
		ClassPriors:     make(map[string]float64),
		WordFrequencies: make(map[string]map[string]int),
		WordProbs:       make(map[string]map[string]float64),
		NGramProbs:      make(map[string]map[string]float64),
		Rules:           loadSpamRules(),
		Version:         VERSION,
		LastUpdated:     time.Now(),
	}

	// Initialize word frequencies
	model.WordFrequencies["spam"] = make(map[string]int)
	model.WordFrequencies["ham"] = make(map[string]int)

	// Initialize word probabilities
	model.WordProbs["spam"] = make(map[string]float64)
	model.WordProbs["ham"] = make(map[string]float64)

	// Initialize n-gram probabilities
	model.NGramProbs["spam"] = make(map[string]float64)
	model.NGramProbs["ham"] = make(map[string]float64)

	// Open the CSV file
	file, err := os.Open(filePath)
	if err != nil {
		logger.Printf("Warning: Could not open training file %s: %v", filePath, err)
		logger.Printf("Using fallback training data instead")
		return trainEnhancedFallbackModel()
	}
	defer file.Close()

	// Create a new CSV reader
	csvReader := csv.NewReader(file)

	// Skip header
	_, err = csvReader.Read()
	if err != nil {
		logger.Printf("Warning: Error reading CSV header: %v", err)
		return trainEnhancedFallbackModel()
	}

	// Read all records
	documents := []Document{}
	spamCount := 0
	hamCount := 0

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			logger.Printf("Warning: Error reading CSV: %v", err)
			continue
		}

		if len(record) >= 2 {
			// Process text
			preprocessedText := preprocessText(record[1])

			// Extract features
			features := extractFeatures(preprocessedText, preprocessedText, "", "")

			doc := Document{
				Label:    record[0],
				Text:     preprocessedText,
				Features: features,
			}
			documents = append(documents, doc)

			// Count classes
			if record[0] == "spam" {
				spamCount++
			} else {
				hamCount++
			}
		}
	}

	if len(documents) == 0 {
		logger.Printf("Warning: No training data found in CSV. Using fallback data.")
		return trainEnhancedFallbackModel()
	}

	// Calculate class priors
	totalDocs := float64(spamCount + hamCount)
	model.ClassPriors["spam"] = float64(spamCount) / totalDocs
	model.ClassPriors["ham"] = float64(hamCount) / totalDocs

	// Process all documents to build vocabulary and count word frequencies
	for _, doc := range documents {
		// Add words to vocabulary
		words := strings.Fields(doc.Text)
		for _, word := range words {
			if _, exists := model.Vocabulary[word]; !exists {
				model.Vocabulary[word] = len(model.Vocabulary)
			}
			model.WordFrequencies[doc.Label][word]++
		}

		// Process n-grams (bigrams)
		bigrams := extractNGrams(doc.Text, 2)
		for _, bigram := range bigrams {
			model.NGramProbs[doc.Label][bigram]++
		}
	}

	// Calculate word probabilities with Laplace smoothing
	vocabSize := float64(len(model.Vocabulary))
	totalSpamWords := sumValues(model.WordFrequencies["spam"])
	totalHamWords := sumValues(model.WordFrequencies["ham"])

	for word := range model.Vocabulary {
		// Calculate spam probability
		spamCount := float64(model.WordFrequencies["spam"][word])
		model.WordProbs["spam"][word] = (spamCount + 1) / (float64(totalSpamWords) + vocabSize)

		// Calculate ham probability
		hamCount := float64(model.WordFrequencies["ham"][word])
		model.WordProbs["ham"][word] = (hamCount + 1) / (float64(totalHamWords) + vocabSize)
	}

	// Normalize n-gram probabilities
	normalizeMapValues(model.NGramProbs["spam"])
	normalizeMapValues(model.NGramProbs["ham"])

	// Update model metadata
	model.TrainingSize = len(documents)
	model.FeatureCount = len(model.Vocabulary)

	// Save the model
	saveModel(model, modelPath)

	fmt.Printf("Trained enhanced model with %d documents (%d spam, %d ham)\n",
		len(documents), spamCount, hamCount)

	return model
}

// loadModelIfExists loads a model from a file if it exists
func loadModelIfExists(modelPath string) *EnhancedModel {
	modelFile, err := ioutil.ReadFile(modelPath)
	if err != nil {
		return nil
	}

	var model EnhancedModel
	err = json.Unmarshal(modelFile, &model)
	if err != nil {
		logger.Printf("Error parsing model file: %v", err)
		return nil
	}

	// Compile rule regexes with error handling
	for i := range model.Rules {
		var err error
		model.Rules[i].Regex, err = regexp.Compile(model.Rules[i].Pattern)
		if err != nil {
			logger.Printf("Warning: Failed to compile regex pattern '%s': %v", model.Rules[i].Pattern, err)
			model.Rules[i].IsActive = false // Disable rules with invalid patterns
		}
	}

	return &model
}

// saveModel saves a model to a file
func saveModel(model *EnhancedModel, modelPath string) {
	// Clear regex field before saving (can't be serialized)
	for i := range model.Rules {
		model.Rules[i].Regex = nil
	}

	modelFile, err := json.MarshalIndent(model, "", "    ")
	if err != nil {
		logger.Printf("Failed to serialize model: %v", err)
		return
	}

	err = ioutil.WriteFile(modelPath, modelFile, 0644)
	if err != nil {
		logger.Printf("Failed to write model file: %v", err)
		return
	}

	// Recompile regexes after saving
	for i := range model.Rules {
		model.Rules[i].Regex, _ = regexp.Compile(model.Rules[i].Pattern)
	}

	logger.Printf("Saved model with %d features to %s", model.FeatureCount, modelPath)
}

// loadSpamRules loads the enhanced spam detection rules
func loadSpamRules() []Rule {
	rules := []Rule{
		{
			Name:        "urgent_action_required",
			Pattern:     `(?i)(urgent|immediate).{0,30}(action|response|attention)`,
			Weight:      0.75,
			Description: "Phrases suggesting urgent action required",
			IsActive:    true,
		},
		{
			Name:        "foreign_money_transfer",
			Pattern:     `(?i)(million|billion).{0,20}(dollar|euro|pound|USD|EUR).{0,30}(transfer|wire|send|receive)`,
			Weight:      0.95,
			Description: "References to transferring large sums of money",
			IsActive:    true,
		},
		{
			Name:        "account_verification",
			Pattern:     `(?i)(verify|confirm|update).{0,20}(account|information|details)`,
			Weight:      0.6,
			Description: "Requests to verify or update account details",
			IsActive:    true,
		},
		{
			Name:        "suspicious_links",
			Pattern:     `(?i)https?://(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]/\w{4,}`,
			Weight:      0.4,
			Description: "Links to potentially suspicious websites",
			IsActive:    true,
		},
		{
			Name:        "prize_winner",
			Pattern:     `(?i)(congratulations|won|winner).{0,40}(prize|lottery|award|cash)`,
			Weight:      0.85,
			Description: "Claims about winning prizes or lotteries",
			IsActive:    true,
		},
		{
			Name:        "banking_alert",
			Pattern:     `(?i)(account|bank).{0,20}(suspend|disabled|locked)`,
			Weight:      0.75,
			Description: "Banking alerts about account issues",
			IsActive:    true,
		},
		{
			Name:        "investment_opportunity",
			Pattern:     `(?i)(investment|opportunity|profit).{0,30}(guaranteed|limited time|exclusive)`,
			Weight:      0.8,
			Description: "Investment opportunities with guarantees",
			IsActive:    true,
		},
		{
			Name:        "pharmaceutical",
			Pattern:     `(?i)(viagra|cialis|pharmacy|prescription|meds|drugs).{0,30}(online|cheap|discount|buy)`,
			Weight:      0.9,
			Description: "Pharmaceutical or medication sales",
			IsActive:    true,
		},
		{
			Name:        "dating_adult",
			Pattern:     `(?i)(sexy|hot|adult|dating).{0,20}(singles|women|men|profiles|pics)`,
			Weight:      0.85,
			Description: "Adult content or dating services",
			IsActive:    true,
		},
		{
			Name:        "caps_excessive",
			Pattern:     `[A-Z]{10,}`,
			Weight:      0.5,
			Description: "Excessive use of capital letters",
			IsActive:    true,
		},
		{
			Name:        "work_from_home",
			Pattern:     `(?i)(work|earn|income).{0,20}(home|anywhere|remotely)`,
			Weight:      0.6,
			Description: "Work from home opportunities",
			IsActive:    true,
		},
		{
			Name:        "crypto_investment",
			Pattern:     `(?i)(bitcoin|crypto|blockchain).{0,30}(invest|profit|earn|trading)`,
			Weight:      0.75,
			Description: "Cryptocurrency investment offers",
			IsActive:    true,
		},
		{
			Name:        "refund_notification",
			Pattern:     `(?i)(refund|rebate|cashback).{0,30}(claim|available|waiting)`,
			Weight:      0.7,
			Description: "Unexpected refund notifications",
			IsActive:    true,
		},
		{
			Name:        "zip_compressed_attachment",
			Pattern:     `(?i)\.zip|\.rar|\.7z`,
			Weight:      0.5,
			Description: "Compressed file attachments",
			IsActive:    true,
		},
		{
			Name:        "suspiciously_similar_domain",
			Pattern:     `(?i)(paypa[l1]|amaz[o0]n|g[o0]{2}gle|faceb[o0]{2}k|micr[o0]s[o0]ft)`,
			Weight:      0.85,
			Description: "Domains similar to legitimate companies",
			IsActive:    true,
		},
	}

	// Compile regexes with error handling
	for i := range rules {
		var err error
		rules[i].Regex, err = regexp.Compile(rules[i].Pattern)
		if err != nil {
			logger.Printf("Warning: Failed to compile regex pattern '%s': %v", rules[i].Pattern, err)
			rules[i].IsActive = false // Disable rules with invalid patterns
		}
	}

	return rules
}

// trainEnhancedFallbackModel is an enhanced fallback model if CSV training fails
func trainEnhancedFallbackModel() *EnhancedModel {
	// Sample training data
	spamExamples := []string{
		"Congratulations! You've won a $1000 gift card. Claim now at our website!",
		"URGENT: Your account has been suspended. Verify your details immediately.",
		"Double your income working from home! Limited time offer - sign up today!",
		"Hot singles in your area want to meet you. Click here for profiles.",
		"Buy medication online without prescription. Lowest prices guaranteed.",
		"You have been selected for an exclusive business opportunity. Respond now!",
		"Claim your unclaimed inheritance of $4.5 million. Contact our agent ASAP.",
		"Invest in Bitcoin now! 500% returns guaranteed in just 2 weeks.",
		"Your PayPal account needs verification. Click the link to update your information.",
		"WINNER ANNOUNCEMENT: You won $5,000,000 in our lottery. Contact to claim.",
	}

	hamExamples := []string{
		"Meeting scheduled for tomorrow at 10 AM. Please bring your reports.",
		"Your package will be delivered today. Tracking number: TN123456789.",
		"Thank you for your recent purchase. Here is your receipt.",
		"The project deadline has been extended to next Friday.",
		"Your monthly account statement is now available.",
		"Reminder: Dentist appointment on Wednesday at 2 PM.",
		"Please review the attached document and provide feedback.",
		"Your flight reservation has been confirmed. Details attached.",
		"Team lunch is scheduled for Friday at the Italian restaurant.",
		"System maintenance scheduled for this weekend. Expect brief downtime.",
	}

	// Create a basic model
	model := &EnhancedModel{
		Vocabulary:      make(map[string]int),
		FeatureWeights:  make(map[string]float64),
		SpamThreshold:   0.75,
		ClassPriors:     make(map[string]float64),
		WordFrequencies: make(map[string]map[string]int),
		WordProbs:       make(map[string]map[string]float64),
		NGramProbs:      make(map[string]map[string]float64),
		Rules:           loadSpamRules(),
		Version:         VERSION,
		LastUpdated:     time.Now(),
	}

	// Initialize word frequencies
	model.WordFrequencies["spam"] = make(map[string]int)
	model.WordFrequencies["ham"] = make(map[string]int)

	// Initialize word probabilities
	model.WordProbs["spam"] = make(map[string]float64)
	model.WordProbs["ham"] = make(map[string]float64)

	// Initialize n-gram probabilities
	model.NGramProbs["spam"] = make(map[string]float64)
	model.NGramProbs["ham"] = make(map[string]float64)

	// Process spam examples
	for _, example := range spamExamples {
		processExampleForModel(model, example, "spam")
	}

	// Process ham examples
	for _, example := range hamExamples {
		processExampleForModel(model, example, "ham")
	}

	// Set class priors
	totalDocs := float64(len(spamExamples) + len(hamExamples))
	model.ClassPriors["spam"] = float64(len(spamExamples)) / totalDocs
	model.ClassPriors["ham"] = float64(len(hamExamples)) / totalDocs

	// Calculate word probabilities with Laplace smoothing
	vocabSize := float64(len(model.Vocabulary))
	totalSpamWords := sumValues(model.WordFrequencies["spam"])
	totalHamWords := sumValues(model.WordFrequencies["ham"])

	for word := range model.Vocabulary {
		// Calculate spam probability
		spamCount := float64(model.WordFrequencies["spam"][word])
		model.WordProbs["spam"][word] = (spamCount + 1) / (float64(totalSpamWords) + vocabSize)

		// Calculate ham probability
		hamCount := float64(model.WordFrequencies["ham"][word])
		model.WordProbs["ham"][word] = (hamCount + 1) / (float64(totalHamWords) + vocabSize)
	}

	// Normalize n-gram probabilities
	normalizeMapValues(model.NGramProbs["spam"])
	normalizeMapValues(model.NGramProbs["ham"])

	// Update model metadata
	model.TrainingSize = len(spamExamples) + len(hamExamples)
	model.FeatureCount = len(model.Vocabulary)

	fmt.Printf("Created fallback model with %d examples (%d spam, %d ham)\n",
		model.TrainingSize, len(spamExamples), len(hamExamples))

	return model
}

// processExampleForModel processes a single example for the model
func processExampleForModel(model *EnhancedModel, text, label string) {
	// Preprocess the text
	preprocessedText := preprocessText(text)

	// Add words to vocabulary and count frequencies
	words := strings.Fields(preprocessedText)
	for _, word := range words {
		if _, exists := model.Vocabulary[word]; !exists {
			model.Vocabulary[word] = len(model.Vocabulary)
		}
		model.WordFrequencies[label][word]++
	}

	// Process n-grams (bigrams)
	bigrams := extractNGrams(preprocessedText, 2)
	for _, bigram := range bigrams {
		model.NGramProbs[label][bigram]++
	}
}

// sumValues sums the values in a map
func sumValues(m map[string]int) int {
	sum := 0
	for _, v := range m {
		sum += v
	}
	return sum
}

// normalizeMapValues normalizes the values in a map to sum to 1
func normalizeMapValues(m map[string]float64) {
	sum := 0.0
	for _, v := range m {
		sum += v
	}

	if sum > 0 {
		for k := range m {
			m[k] /= sum
		}
	}
}

// evaluateRules evaluates the rules against email content
func evaluateRules(model *EnhancedModel, text string) (float64, []string) {
	totalWeight := 0.0
	maxWeight := 0.0
	triggeredRules := []string{}

	for _, rule := range model.Rules {
		if !rule.IsActive || rule.Regex == nil {
			continue
		}

		if rule.Regex.MatchString(text) {
			triggeredRules = append(triggeredRules, rule.Name)
			totalWeight += rule.Weight
			if rule.Weight > maxWeight {
				maxWeight = rule.Weight
			}
		}
	}

	// Normalize score between 0 and 1
	ruleScore := 0.0
	if len(triggeredRules) > 0 {
		// Use a combination of max weight and average weight
		avgWeight := totalWeight / float64(len(triggeredRules))
		ruleScore = 0.7*maxWeight + 0.3*avgWeight

		// Adjust for the number of triggered rules
		ruleScore = math.Min(ruleScore*(1.0+0.1*float64(len(triggeredRules)-1)), 1.0)
	}

	return ruleScore, triggeredRules
}

// classifyEmailEnhanced classifies an email using the enhanced model
func classifyEmailEnhanced(subject, body, from, headers string, model *EnhancedModel) (string, float64, []string, map[string]float64) {
	// Make sure we have some text to work with
	if subject == "" && body == "" {
		// Return safe defaults if no text is provided
		return "Ham Mail (Not Spam)", 0.0, []string{}, map[string]float64{
			"naive_bayes_score": 0.0,
			"rule_based_score":  0.0,
			"combined_score":    0.0,
		}
	}

	// Start with rule-based classification
	fullText := subject + " " + body
	ruleScore, triggeredRules := evaluateRules(model, fullText)

	// Extract features
	features := extractFeatures(subject, body, from, headers)

	// Preprocess text for ML model
	preprocessedText := preprocessText(fullText)

	// Calculate ML probabilities using Naive Bayes
	words := strings.Fields(preprocessedText)

	// Calculate log probabilities to avoid underflow
	logSpamProb := math.Log(model.ClassPriors["spam"])
	logHamProb := math.Log(model.ClassPriors["ham"])

	for _, word := range words {
		if spamProb, exists := model.WordProbs["spam"][word]; exists {
			logSpamProb += math.Log(spamProb)
		}

		if hamProb, exists := model.WordProbs["ham"][word]; exists {
			logHamProb += math.Log(hamProb)
		}
	}

	// Process n-grams (bigrams)
	bigrams := extractNGrams(preprocessedText, 2)
	for _, bigram := range bigrams {
		if spamProb, exists := model.NGramProbs["spam"][bigram]; exists && spamProb > 0 {
			logSpamProb += math.Log(spamProb)
		}

		if hamProb, exists := model.NGramProbs["ham"][bigram]; exists && hamProb > 0 {
			logHamProb += math.Log(hamProb)
		}
	}

	// Convert log probabilities back to standard probabilities
	spamProb := 0.0
	if logSpamProb > logHamProb {
		spamProb = 1.0 / (1.0 + math.Exp(logHamProb-logSpamProb))
	} else {
		spamProb = 1.0 - 1.0/(1.0+math.Exp(logSpamProb-logHamProb))
	}

	// Combine rule-based and ML scores
	combinedScore := 0.7*spamProb + 0.3*ruleScore

	// Apply feature adjustments
	if features["is_whitelisted"] > 0 {
		combinedScore *= 0.5 // Reduce score for whitelisted domains
	}

	if features["is_reply_or_forward"] > 0 {
		combinedScore *= 0.7 // Reduce score for replies/forwards
	}

	if features["phishing_url_count"] > 0 {
		combinedScore = math.Min(combinedScore*1.5, 0.99) // Increase for phishing URLs
	}

	if features["blacklisted_word_count"] > 3 {
		combinedScore = math.Min(combinedScore*1.3, 0.99) // Increase for blacklisted words
	}

	// Create classifier results map for detailed reporting
	classifierResults := map[string]float64{
		"naive_bayes_score": spamProb,
		"rule_based_score":  ruleScore,
		"combined_score":    combinedScore,
	}

	// Final classification
	classification := "Ham Mail (Not Spam)"
	if combinedScore >= model.SpamThreshold {
		classification = "Spam Mail"
	}

	return classification, combinedScore, triggeredRules, classifierResults
}

// extractEmailHeader extracts header information from an email
func extractEmailHeader(header gomail.Header) (HeaderAnalysis, error) {
	analysis := HeaderAnalysis{}

	// Check SPF
	receivedSPF := header.Get("Received-SPF")
	if receivedSPF != "" {
		analysis.HasSPF = true
		analysis.SPFResult = extractSPFResult(receivedSPF)
	}

	// Check DKIM
	authResults := header.Get("Authentication-Results")
	if strings.Contains(authResults, "dkim=") {
		analysis.HasDKIM = true
		analysis.DKIMValid = strings.Contains(authResults, "dkim=pass") ||
			strings.Contains(authResults, "dkim=valid")
	}

	// Check DMARC
	if strings.Contains(authResults, "dmarc=") {
		analysis.HasDMARC = true
		analysis.DMARCPass = strings.Contains(authResults, "dmarc=pass")
	}

	return analysis, nil
}

// extractSPFResult extracts the result from an SPF header
func extractSPFResult(spfHeader string) string {
	if strings.Contains(spfHeader, "pass") {
		return "pass"
	} else if strings.Contains(spfHeader, "fail") {
		return "fail"
	} else if strings.Contains(spfHeader, "softfail") {
		return "softfail"
	} else if strings.Contains(spfHeader, "neutral") {
		return "neutral"
	} else if strings.Contains(spfHeader, "none") {
		return "none"
	} else {
		return "unknown"
	}
}

// extractEnhancedEmailContent extracts content and metadata from an email message
func extractEnhancedEmailContent(msg *gomail.Reader) (string, string, HeaderAnalysis, bool, int, int, error) {
	// Get the header
	header := msg.Header
	subject := header.Get("Subject")
	var bodyPlain, bodyHTML strings.Builder

	// Extract header analysis
	headerAnalysis, err := extractEmailHeader(header)
	if err != nil {
		logger.Printf("Error analyzing headers: %v", err)
	}

	// Initialize counters
	hasAttachments := false
	numLinks := 0
	numImages := 0

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

			// Check for attachments
			contentDisposition, _, _ := h.ContentDisposition()
			if contentDisposition == "attachment" {
				hasAttachments = true
				continue
			}

			// Process text parts
			if strings.HasPrefix(contentType, "text/plain") {
				buf := new(strings.Builder)
				_, err := io.Copy(buf, part.Body)
				if err != nil {
					continue
				}
				bodyPlain.WriteString(buf.String())
			} else if strings.HasPrefix(contentType, "text/html") {
				buf := new(strings.Builder)
				_, err := io.Copy(buf, part.Body)
				if err != nil {
					continue
				}

				htmlContent := buf.String()
				bodyHTML.WriteString(htmlContent)

				// Count links in HTML
				linkRegex := regexp.MustCompile(`<a\s+[^>]*href=["']([^"']+)["']`)
				linkMatches := linkRegex.FindAllStringSubmatch(htmlContent, -1)
				numLinks += len(linkMatches)

				// Count images in HTML
				imgRegex := regexp.MustCompile(`<img\s+[^>]*src=["']([^"']+)["']`)
				imgMatches := imgRegex.FindAllStringSubmatch(htmlContent, -1)
				numImages += len(imgMatches)
			}
		}
	}

	// Use HTML body if plain text is empty
	var body string
	if bodyPlain.Len() > 0 {
		body = bodyPlain.String()
	} else if bodyHTML.Len() > 0 {
		// Simple HTML to text conversion
		htmlBody := bodyHTML.String()
		re := regexp.MustCompile(`<[^>]*>`)
		body = re.ReplaceAllString(htmlBody, " ")
		body = html.UnescapeString(body)
	}

	return subject, body, headerAnalysis, hasAttachments, numLinks, numImages, nil
}

// scanEmailsForSpam connects to an email server and scans for spam with enhanced methods
func scanEmailsForSpam(emailProvider, emailAddress, password string, config Config, model *EnhancedModel) []EmailResult {
	defaultServers := map[string]string{
		"gmail":   "imap.gmail.com",
		"outlook": "outlook.office365.com",
		"yahoo":   "imap.mail.yahoo.com",
		"aol":     "imap.aol.com",
		"zoho":    "imap.zoho.com",
		"proton":  "imap.protonmail.ch",
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
	resultChan := make(chan EmailResult, config.MaxEmailsToScan)
	var wg sync.WaitGroup

	// Create a semaphore to limit concurrent workers
	semaphore := make(chan struct{}, config.ConcurrentWorkers)

	// Get list of available mailboxes
	mailboxes := make(chan *imap.MailboxInfo, 10)
	done := make(chan error, 1)
	go func() {
		done <- c.List("", "*", mailboxes)
	}()

	availableFolders := make(map[string]bool)
	for m := range mailboxes {
		availableFolders[m.Name] = true
	}

	if err := <-done; err != nil {
		fmt.Printf("Error listing folders: %v\n", err)
	}

	// Add a special case for Gmail's All Mail folder
	if emailProvider == "gmail" {
		availableFolders["[Gmail]/All Mail"] = true
	}

	// Process each folder
	for _, folder := range config.ScanFolders {
		// Check if folder exists
		if !availableFolders[folder] {
			// For Gmail, try with [Gmail] prefix
			if emailProvider == "gmail" && availableFolders["[Gmail]/"+folder] {
				folder = "[Gmail]/" + folder
			} else {
				fmt.Printf("Folder not found: %s\n", folder)
				continue
			}
		}

		_, err := c.Select(folder, false)
		if err != nil {
			fmt.Printf("Could not access folder: %s (%v)\n", folder, err)
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

		// Create a progress indicator
		progress := make(chan int, 10)
		go func() {
			total := len(uids)
			current := 0
			for inc := range progress {
				current += inc
				fmt.Printf("\rProcessed %d/%d emails (%.1f%%)", current, total, float64(current)/float64(total)*100)
			}
			fmt.Println()
		}()

		// Divide UIDs into smaller batches for efficiency
		batchSize := 10
		for i := 0; i < len(uids); i += batchSize {
			end := i + batchSize
			if end > len(uids) {
				end = len(uids)
			}

			batchUIDs := uids[i:end]

			seqSet := new(imap.SeqSet)
			seqSet.AddNum(batchUIDs...)

			// Get the whole message body
			section := &imap.BodySectionName{}
			items := []imap.FetchItem{section.FetchItem(), imap.FetchEnvelope, imap.FetchRFC822Size}

			messages := make(chan *imap.Message, 10)
			done := make(chan error, 1)
			go func() {
				done <- c.Fetch(seqSet, items, messages)
			}()

			// Process messages in parallel
			for msg := range messages {
				wg.Add(1)
				semaphore <- struct{}{} // Acquire semaphore

				go func(msg *imap.Message) {
					defer wg.Done()
					defer func() { <-semaphore }() // Release semaphore

					startTime := time.Now()

					r := msg.GetBody(section)
					if r == nil {
						progress <- 1
						return
					}

					// Get message size
					var messageSize int
					if msg.Size > 0 {
						messageSize = int(msg.Size)
					}

					// Parse message
					mr, err := gomail.CreateReader(r)
					if err != nil {
						progress <- 1
						return
					}

					// Extract enhanced content
					subject, body, headerAnalysis, hasAttachments, numLinks, numImages, err := extractEnhancedEmailContent(mr)
					if err != nil {
						progress <- 1
						return
					}

					// Check cache if enabled
					if config.UseCache {
						emailHash := generateEmailHash(subject, body, msg.Envelope.Date.String())
						if cachedResult, found := cache.EmailHashes[emailHash]; found {
							resultChan <- cachedResult
							progress <- 1
							return
						}
					}

					// Get header as string for analysis
					// Fixed: Generate header string from envelope fields instead
					var headerBuf bytes.Buffer
					if msg.Envelope != nil {
						fmt.Fprintf(&headerBuf, "From: %s\r\n", formatAddress(msg.Envelope.From))
						fmt.Fprintf(&headerBuf, "To: %s\r\n", formatAddress(msg.Envelope.To))
						fmt.Fprintf(&headerBuf, "Subject: %s\r\n", msg.Envelope.Subject)
						fmt.Fprintf(&headerBuf, "Date: %s\r\n", msg.Envelope.Date.String())
					}
					headerStr := headerBuf.String()

					// Classify the email
					classification, spamScore, triggeredRules, classifierResults :=
						classifyEmailEnhanced(subject, body, formatAddress(msg.Envelope.From), headerStr, model)

					// Calculate processing time
					processingTime := float64(time.Since(startTime).Microseconds()) / 1000.0

					// Store results
					from := formatAddress(msg.Envelope.From)
					date := ""
					if msg.Envelope != nil && msg.Envelope.Date != (time.Time{}) {
						date = msg.Envelope.Date.String()
					}

					// Extract features for reporting
					features := extractFeatures(subject, body, from, headerStr)

					result := EmailResult{
						Subject:           subject,
						From:              from,
						Date:              date,
						Classification:    classification,
						SpamScore:         spamScore,
						Confidence:        spamScore,
						Features:          features,
						Rules:             triggeredRules,
						HTMLPresent:       strings.Contains(body, "<html") || strings.Contains(body, "<body"),
						NumLinks:          numLinks,
						NumImages:         numImages,
						HasAttachments:    hasAttachments,
						HeaderAnalysis:    headerAnalysis,
						ProcessingTime:    processingTime,
						RawSize:           messageSize,
						ClassifierResults: classifierResults,
					}

					resultChan <- result

					// Update statistics
					stats.Add(classification == "Spam Mail", processingTime, spamScore)

					// Update cache if enabled
					if config.UseCache {
						emailHash := generateEmailHash(subject, body, date)
						cache.EmailHashes[emailHash] = result
						cache.Size = len(cache.EmailHashes)
						cache.LastUpdate = time.Now()
					}

					progress <- 1
				}(msg)
			}

			if err := <-done; err != nil {
				fmt.Printf("Error fetching messages: %v\n", err)
			}
		}

		close(progress)
		fmt.Printf("Completed scanning folder: %s\n", folder)
	}

	// Wait for all goroutines to finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for result := range resultChan {
		results = append(results, result)
	}

	return results
}

// formatAddress formats an address from imap envelope
func formatAddress(addresses []*imap.Address) string {
	if len(addresses) == 0 {
		return ""
	}
	addr := addresses[0]
	if addr == nil {
		return ""
	}
	return fmt.Sprintf("%s@%s", addr.MailboxName, addr.HostName)
}

// displayEnhancedResults displays the enhanced scan results
func displayEnhancedResults(results []EmailResult, config Config) {
	if len(results) == 0 {
		fmt.Println("No emails were processed.")
		return
	}

	// Sort results by classification and spam score
	sort.Slice(results, func(i, j int) bool {
		if results[i].Classification != results[j].Classification {
			return results[i].Classification == "Spam Mail"
		}
		return results[i].SpamScore > results[j].SpamScore
	})

	// Calculate summary statistics
	spamCount := 0
	totalConfidence := 0.0
	highConfidenceSpam := 0
	mediumConfidenceSpam := 0
	lowConfidenceSpam := 0

	for _, r := range results {
		if r.Classification == "Spam Mail" {
			spamCount++
			totalConfidence += r.SpamScore

			if r.SpamScore >= 0.9 {
				highConfidenceSpam++
			} else if r.SpamScore >= 0.75 {
				mediumConfidenceSpam++
			} else {
				lowConfidenceSpam++
			}
		}
	}

	hamCount := len(results) - spamCount
	avgConfidence := 0.0
	if spamCount > 0 {
		avgConfidence = totalConfidence / float64(spamCount)
	}

	// Print summary
	fmt.Println("\n" + strings.Repeat("=", 60))
	color.New(color.FgCyan).Add(color.Bold).Println("SCAN RESULTS")
	fmt.Println(strings.Repeat("=", 60))

	fmt.Printf("Total emails scanned: %d\n", len(results))

	spamPerc := getPercentage(spamCount, len(results))
	hamPerc := getPercentage(hamCount, len(results))

	// Use color for better visibility
	spam := color.New(color.FgRed).SprintFunc()
	ham := color.New(color.FgGreen).SprintFunc()

	fmt.Printf("Spam emails detected: %s (%d - %.1f%%)\n", spam(spamCount), spamCount, spamPerc)
	fmt.Printf("Legitimate emails: %s (%d - %.1f%%)\n", ham(hamCount), hamCount, hamPerc)

	// Display confidence levels
	if spamCount > 0 {
		fmt.Printf("\nSpam confidence levels:\n")
		fmt.Printf("  High confidence (>90%%): %d (%.1f%%)\n", highConfidenceSpam, getPercentage(highConfidenceSpam, spamCount))
		fmt.Printf("  Medium confidence (75-90%%): %d (%.1f%%)\n", mediumConfidenceSpam, getPercentage(mediumConfidenceSpam, spamCount))
		fmt.Printf("  Low confidence (<75%%): %d (%.1f%%)\n", lowConfidenceSpam, getPercentage(lowConfidenceSpam, spamCount))
		fmt.Printf("  Average confidence: %.1f%%\n", avgConfidence*100)
	}

	// Display spam emails
	if spamCount > 0 {
		fmt.Println("\n" + strings.Repeat("-", 60))
		color.New(color.FgRed).Add(color.Bold).Println("POTENTIAL SPAM EMAILS")
		fmt.Println(strings.Repeat("-", 60))

		// Fixed: Use _ instead of i since i was unused
		for _, r := range results {
			if r.Classification == "Spam Mail" {
				// Color based on confidence
				var confidenceColor func(a ...interface{}) string
				if r.SpamScore >= 0.9 {
					confidenceColor = color.New(color.FgRed).Add(color.Bold).SprintFunc()
				} else if r.SpamScore >= 0.75 {
					confidenceColor = color.New(color.FgYellow).SprintFunc()
				} else {
					confidenceColor = color.New(color.FgHiYellow).SprintFunc()
				}

				fmt.Printf("Subject: %s\n", confidenceColor(r.Subject))
				fmt.Printf("   From: %s\n", r.From)
				fmt.Printf("   Date: %s\n", r.Date)
				fmt.Printf("   Spam Score: %.1f%% confidence\n", r.SpamScore*100)

				// Show top triggered rules
				if len(r.Rules) > 0 {
					fmt.Print("   Triggered Rules: ")
					for j, rule := range r.Rules {
						if j > 0 {
							fmt.Print(", ")
						}
						if j >= 3 {
							fmt.Printf("and %d more", len(r.Rules)-3)
							break
						}
						fmt.Print(rule)
					}
					fmt.Println()
				}

				// Show key features
				fmt.Printf("   Features: ")
				featuresShown := 0
				if r.Features["url_count"] > 0 {
					fmt.Printf("URLs: %d", int(r.Features["url_count"]))
					featuresShown++
				}
				if r.Features["suspicious_phrase_count"] > 0 {
					if featuresShown > 0 {
						fmt.Print(", ")
					}
					fmt.Printf("Suspicious phrases: %d", int(r.Features["suspicious_phrase_count"]))
					featuresShown++
				}
				if r.Features["urgency_count"] > 0 {
					if featuresShown > 0 {
						fmt.Print(", ")
					}
					fmt.Printf("Urgency indicators: %d", int(r.Features["urgency_count"]))
					featuresShown++
				}
				fmt.Println()

				fmt.Println(strings.Repeat("-", 60))
			}
		}
	}
}

// getPercentage calculates a percentage safely
func getPercentage(part, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}

// saveEnhancedResults saves the results to a JSON file with enhanced reporting
func saveEnhancedResults(results []EmailResult, emailAddress string, config Config) {
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

	color.New(color.FgGreen).Printf("\nResults saved to: %s\n", filename)

	// Generate HTML report if enabled
	if config.EnableReporting {
		htmlFilename := filepath.Join(config.ResultsDirectory, fmt.Sprintf("report_%s_%s.html", emailUser, timestamp))
		generateHTMLReport(results, htmlFilename)
	}
}

// generateHTMLReport generates an HTML report from the results
func generateHTMLReport(results []EmailResult, filename string) {
	// Count statistics
	spamCount := 0
	hamCount := 0
	for _, r := range results {
		if r.Classification == "Spam Mail" {
			spamCount++
		} else {
			hamCount++
		}
	}

	// Start HTML content
	html := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Spam Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #3498db; margin-top: 30px; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .spam-email { background-color: #fff; border-left: 4px solid #e74c3c; padding: 15px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .ham-email { background-color: #fff; border-left: 4px solid #2ecc71; padding: 15px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .confidence { display: inline-block; padding: 3px 6px; border-radius: 3px; font-size: 12px; font-weight: bold; }
        .high { background-color: #e74c3c; color: white; }
        .medium { background-color: #f39c12; color: white; }
        .low { background-color: #f1c40f; color: #333; }
        .rules { margin-top: 10px; font-size: 14px; color: #7f8c8d; }
        .feature { display: inline-block; background-color: #eee; padding: 2px 5px; margin: 2px; border-radius: 3px; font-size: 12px; }
        .meta { color: #7f8c8d; font-size: 14px; margin-top: 5px; }
        .chart { margin-top: 30px; }
    </style>
</head>
<body>
    <h1>Email Spam Scan Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total emails scanned: <strong>` + strconv.Itoa(len(results)) + `</strong></p>
        <p>Spam emails detected: <strong>` + strconv.Itoa(spamCount) + ` (` + fmt.Sprintf("%.1f%%", getPercentage(spamCount, len(results))) + `)</strong></p>
        <p>Legitimate emails: <strong>` + strconv.Itoa(hamCount) + ` (` + fmt.Sprintf("%.1f%%", getPercentage(hamCount, len(results))) + `)</strong></p>
        <p>Scan date: <strong>` + time.Now().Format("Jan 2, 2006 15:04:05") + `</strong></p>
    </div>
`

	// Sort results by classification and score
	sort.Slice(results, func(i, j int) bool {
		if results[i].Classification != results[j].Classification {
			return results[i].Classification == "Spam Mail"
		}
		return results[i].SpamScore > results[j].SpamScore
	})

	// Add spam emails section
	if spamCount > 0 {
		html += `    <h2>Potential Spam Emails</h2>
`

		for _, r := range results {
			if r.Classification == "Spam Mail" {
				confidenceClass := "low"
				if r.SpamScore >= 0.9 {
					confidenceClass = "high"
				} else if r.SpamScore >= 0.75 {
					confidenceClass = "medium"
				}

				html += `    <div class="spam-email">
        <h3>` + r.Subject + `</h3>
        <p>From: <strong>` + r.From + `</strong></p>
        <p>Date: ` + r.Date + `</p>
        <p>Spam Score: <span class="confidence ` + confidenceClass + `">` + fmt.Sprintf("%.1f%%", r.SpamScore*100) + `</span></p>
`

				// Add rules if available
				if len(r.Rules) > 0 {
					html += `        <div class="rules">Triggered rules: `
					for j, rule := range r.Rules {
						if j > 0 {
							html += `, `
						}
						html += rule
					}
					html += `</div>
`
				}

				// Add features
				html += `        <div class="features">`
				if r.Features["url_count"] > 0 {
					html += `<span class="feature">URLs: ` + fmt.Sprintf("%d", int(r.Features["url_count"])) + `</span>`
				}
				if r.Features["suspicious_phrase_count"] > 0 {
					html += `<span class="feature">Suspicious phrases: ` + fmt.Sprintf("%d", int(r.Features["suspicious_phrase_count"])) + `</span>`
				}
				if r.Features["urgency_count"] > 0 {
					html += `<span class="feature">Urgency indicators: ` + fmt.Sprintf("%d", int(r.Features["urgency_count"])) + `</span>`
				}
				if r.NumLinks > 0 {
					html += `<span class="feature">Links: ` + strconv.Itoa(r.NumLinks) + `</span>`
				}
				if r.NumImages > 0 {
					html += `<span class="feature">Images: ` + strconv.Itoa(r.NumImages) + `</span>`
				}
				if r.HasAttachments {
					html += `<span class="feature">Has attachments</span>`
				}
				html += `</div>
    </div>
`
			}
		}
	}

	// Close HTML
	html += `</body>
</html>`

	// Write to file
	err := ioutil.WriteFile(filename, []byte(html), 0644)
	if err != nil {
		fmt.Printf("Error writing HTML report: %v\n", err)
		return
	}

	color.New(color.FgGreen).Printf("HTML report saved to: %s\n", filename)
}
