import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    
    # Part-of-speech tagging
    words = pos_tag(words)
    
    # Extract only nouns and adjectives
    words = [word[0] for word in words if word[1] in ['NN', 'NNS', 'NNP', 'JJ']]
    
    return ' '.join(words)

def textrank_keyword_extraction(text, num_keywords=5):
    preprocessed_text = preprocess_text(text)

    # Create a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a graph representation
    graph = nx.Graph()
    
    # Add nodes (keywords) to the graph
    for word in feature_names:
        graph.add_node(word)
    
    # Calculate edge weights based on the similarity between words
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            word1 = feature_names[i]
            word2 = feature_names[j]
            similarity = tfidf_matrix[0, i] * tfidf_matrix[0, j]
            if similarity > 0:
                graph.add_edge(word1, word2, weight=similarity)

    # Use the PageRank algorithm (TextRank) to rank keywords
    keyword_ranks = nx.pagerank(graph)

    # Sort keywords by rank
    sorted_keywords = sorted(keyword_ranks.items(), key=lambda x: x[1], reverse=True)

    # Extract the top 'num_keywords' keywords
    top_keywords = [keyword for keyword, rank in sorted_keywords[:num_keywords]]

    return top_keywords

# Sample document
document = """
Natural language processing (NLP) is a field of computer science that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human language in a way that is valuable.
"""

# Extract keywords
keywords = textrank_keyword_extraction(document)
print("Top Keywords:", keywords)
