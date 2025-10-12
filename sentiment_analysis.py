"""
requirement6.py (Word Segmentation)
--------
This is the solution for the word segmentation problem.
This uses dynamic programming.
"""

# --- Core Data Manipulation and NLP ---
import pandas as pd
from nltk.tokenize import sent_tokenize

''' ---------------------- Data Processing ---------------------- '''
# Load exact same data of airline_reviews, but without spaces
# PATH = 'data/airlines_review_no_space.csv'
# df = pd.read_csv(PATH)

def load_afinn_lexicon(afinn_path):
    afinn_dict = {}
    try:
        with open(afinn_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word, score = line.rsplit('\t', 1)
                afinn_dict[word] = int(score)
    except FileNotFoundError:
        print(f"File not found: {afinn_path}")
    return afinn_dict

def replace_missing_value(df):
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna("NA")
    return 

def remove_special_characters(df, removechar, char):
    """
    Function used in -> load_and_clean_data().
    Removes a predefined list of noise characters from specific text columns.

    - Characters are replaced with a single space in the 'Airlines' column 
      to prevent word concatenation.
    - Characters are completely removed (replaced with an empty string) in 
      the 'Text Content' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        removechar (list): A list of strings/characters to search for and remove.
        char (str): An unused parameter (retained from original signature).
        
    Returns:
        pd.DataFrame: The DataFrame with characters removed from specified columns.
    """
    for char in removechar:
        df['Airlines'] = df['Airlines'].str.replace(char, ' ', regex=False)
        df['Text Content'] = df['Text Content'].str.replace(char, '', regex=False)
    return df

def load_and_clean_data(df):
    #df = pd.read_csv(PATH)
    
    # Remove duplicates and NaNs
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Characters to remove
    removechar = [ '@', '#', '$', '%', '^', '&', '*', '(', ')',
                   '-', '_', '=', '+', '{', '}', '[', ']', '|',
                   '\\', ':', ';', '"', "'", '<', '>', '?',
                   '/', '~', '`', '✅ Trip Verified', 'Not Verified',
                   'Â Â', '✅ Verified Review']
    
    # Apply cleaning functions
    replace_missing_value(df)
    remove_special_characters(df, removechar, char='')

    # Standardize text case
    df['Airlines'] = df['Airlines'].str.title()
    df['Name'] = df['Name'].str.title()
    df['Text Content Lower Case'] = df['Text Content'].str.lower()

    df['Airlines'] = df['Airlines'].str.lstrip()
    df['Name'] = df['Name'].str.lstrip()
    if 'Date Published' in df.columns:
        df['Date Published'] = df['Date Published'].str.lstrip()
    df['Text Content'] = df['Text Content'].str.lstrip()
    
    df.to_csv('data/airlines_review_cleaned.csv', index=False)
    return df


''' ---------------------- Sentiment Scoring Algorithm ---------------------- '''

'''Start of Req 6: Word Segmentation using Dynamic Programming'''
def load_dictionary(dict_path):
    """
    Loads a word list from a file into a set for fast lookup.
    The function reads a file line by line, strips whitespace, converts
    each word to lowercase, and adds it to a set.

    Args:
        dict_path (str): The file path to the dictionary file,
                         with one word per line.
    Returns:
        set: A set containing all unique words from the file in lowercase.
    """
    words = set()
    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            words.add(w.lower())
    return words

def segment_text(text, dictionary, max_word_length=30, unknown_penalty=1):
    """
    Segments a string (with no spaces) into a sequence of known words using 
    a Dynamic Programming (DP) approach.

    The DP algorithm finds the highest-scoring segmentation based on word 
    existence in the dictionary and applies a penalty for unknown words.

    Args:
        text (str or None): The concatenated string to be segmented.
        dictionary (set): A set of valid words for lookup (e.g., from load_dictionary).
        max_word_length (int, optional): The maximum length to consider for a word 
                                         segment. Defaults to 30.
        unknown_penalty (int, optional): The score penalty applied for each 
                                         character of an unknown word. Defaults to 1.
    Returns:
        str: The segmented string, with spaces between identified words. 
             Returns the original text if no segmentation can be found.
    """
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""
    
    s_lower = s.lower()
    n = len(s_lower)
    best_score = [-10**9] * (n + 1)
    prev_idx = [-1] * (n + 1)
    best_score[0] = 0
    max_len = min(max_word_length, n)
    length_bias = 0.3
    
    for i in range(1, n + 1):       
        start_j = max(0, i - max_len)
        
        for j in range(start_j, i):
            chunk = s_lower[j:i]
            L = len(chunk)

            if chunk in dictionary:
                multiplier = 1.0 + length_bias * (L - 1)
                score = best_score[j] + (L * multiplier)
            else:
                score = best_score[j] - (L * unknown_penalty)

            if score > best_score[i]:
                best_score[i] = score
                prev_idx[i] = j

    if prev_idx[n] == -1:
        return s

    tokens = []
    i = n
    while i > 0:
        j = prev_idx[i]
        if j == -1:
            tokens.append(s[j:i])
            break
        tokens.append(s[j:i])
        i = j
    tokens.reverse()
    return ' '.join(tokens)

def call_dynamic_prog(df):
    """
    Applies word segmentation to the 'Text Content' column of a DataFrame 
    using Dynamic Programming, but only if unsegmented (concatenated) text is detected.

    The function first loads the word dictionary, determines the optimal segmentation 
    parameters, and then applies the 'segment_text' function to the column, 
    replacing the original content with the segmented text.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the 'Text Content' 
                           column to be segmented.
    Returns:
        pd.DataFrame: The DataFrame with the 'Text Content' column updated 
                      with segmented strings (or unchanged if no segmentation 
                      was necessary).
    """
    # Ensure the column is string type
    df['Text Content'] = df['Text Content'].astype(str)
    
    # Checks if there are any spaces in the column Text Content"
    if df['Text Content'].apply(lambda x: ' ' not in x).any():
        print("⚠️  No spaces detected. Segmentation needed!"
              "\n🚧 Running Word Segmentation using Dynamic Programming..."
              "\n⏱️  This will take a while..")
        
        dict_path = 'data/words.txt'    
        dictionary = load_dictionary(dict_path)         
        
        if dictionary:
            max_word_length = max(len(w) for w in dictionary)
            max_word_length = min(max_word_length, 30)
        else:
            max_word_length = 30

        # Replace the 'Text Content' column with the segmented version
        df['Text Content'] = df['Text Content'].astype(str).apply(
            lambda t: segment_text(t, dictionary, max_word_length=max_word_length, 
                                   unknown_penalty=1)
        )

        df.to_csv('data/airlines_review_analysis.csv', index=False)    
    else:
        print("✅ Segmentation not needed!")
    
    return df
    
    
'''End of Req 6: Word Segmentation using Dynamic Programming'''


def tokenize_sentences(text):
    '''
    Function for Sentence tokanization. This function takes some text and returns a list of sentences. 
    If you dont give it a string, it safely returns an empty list instead of crashing.
    
    Args:
        text (str): The input text to be tokenized.
    Returns:
        list: A list of strings, where each string is a sentence. Returns an empty list if input is not a string.
    
    '''
    if not isinstance(text, str):
        return []
    return sent_tokenize(text)


def calculate_sentiment_score(sentences, afinn_dict):
    '''
    Calculates the raw sentiment score of a given text segment by summing up AFINN scores of individual words.

    Args:
        sentences (str): The text segment (sentence, paragraph, or full text) to score.
        afinn_dict (dict): The AFINN sentiment lexicon (word: score mapping).
    Returns:
        int/float: The total raw sentiment score for the text segment.
    '''
    score = 0
    words = sentences.lower().split()    # breaks down the sentence into individual words
    for word in words:  # goes through each word in the list 
        score += afinn_dict.get(word, 0)    # looks up each word and adds the score
    return score


def normalize_score(score, text_length):
    '''
    Calculates the normalized sentiment score by dividing the raw score by the text length,
    and clamps the result to the range [-1.0, 1.0].

    Args:
        score (int/float): The raw sentiment score calculated by calculate_sentiment_score.
        text_length (int): The number of words in the text segment.
    Returns:
        float: The normalized sentiment score, clamped between -1.0 and 1.0.
    '''
    if text_length == 0:    # sentences with no words will return a score of 0 (preventing an error of dividing by zero)
        return 0
    # normalization to get a score per word, then clamping to [-1, 1]
    normalized = score / text_length    # calculates the average score per word
    return max(-1.0, min(1.0, normalized))  
    # makes sure the score doesn't go over 1.0 and below -1.0. 
    # (if the score is -2.5, this will return -1.0)


def find_extreme_sentences(sentences, afinn_dict):
    '''
    Finds extreme sentences with highest and lowest normalized sentiment scores
    1. Calculates sentiment scores for each sentence using the AFINN lexicon.
    2. Normalizes the sentiment scores based on sentence length.
    3. Identifies and returns the sentences with the highest and lowest normalized scores.
    
    Args:
        sentences (list): A list of strings, where each string is a sentence.
        afinn_dict (dict): The AFINN sentiment lexicon (word: score mapping).
    Returns:
        tuple: A tuple (most_positive, most_negative), where each element is a dictionary
               like {'sentence': str, 'score': float}. Returns (None, None) if the input list is empty.
    '''
    if not sentences:   # to prevent errors if the list is empty
        return None, None

    # initialises an empty list to hold sentences and their scores
    scored_sentences = []
    for sent in sentences:
        score = calculate_sentiment_score(sent, afinn_dict)     # uses the sentiment score function
        normalized_score = normalize_score(score, len(sent.split()))    # uses the normalization function
        scored_sentences.append({'sentence': sent, 'score': normalized_score})

    # find the sentences with max and min normalized scores
    most_positive = max(scored_sentences, key=lambda x: x['score'])
    most_negative = min(scored_sentences, key=lambda x: x['score'])
    
    return most_positive, most_negative


def sliding_window_analysis_words(text, afinn_dict, window_size=10):
    """
    Applies a sliding window to find the most positive and negative text segments,
    based on a word-level window size.
    
    Args:
        text (str): The full text to analyze.
        afinn_dict (dict): The AFINN sentiment lexicon.
        window_size (int): The number of words to include in each window.

    Returns:
        tuple: A tuple containing the most positive and most negative paragraphs.
    """
    if not text:
        return None, None
        
    words = text.lower().split()
    if len(words) < window_size:
        return None, None
    
    scored_windows = []
    # Slide the window across the words
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        paragraph_text = ' '.join(window)
        
        # Calculate score for the paragraph window
        score = calculate_sentiment_score(paragraph_text, afinn_dict)
        normalized_score = normalize_score(score, len(paragraph_text.split()))

        scored_windows.append({'paragraph': paragraph_text, 'score': normalized_score})

    if not scored_windows:
        return None, None

    # Find the paragraphs with max and min scores
    most_positive_paragraph = max(scored_windows, key=lambda x: x['score'])
    most_negative_paragraph = min(scored_windows, key=lambda x: x['score'])
    
    return most_positive_paragraph, most_negative_paragraph


'''---------------------- Applying analysis to the dataframe ---------------------- '''
def run_sentiment_analysis(df, afinn_dict):
    '''
    Applies all sentiment analysis functions (tokenization, scoring, normalization, 
    extreme sentence finding, and sliding window analysis) to a Pandas DataFrame.
    It adds 10 new columns containing all the analysis results.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'Text Content' column.
        afinn_dict (dict): The AFINN sentiment lexicon (word: score mapping).
    Returns:
        pd.DataFrame: The original DataFrame with all sentiment analysis results added as new columns.
    '''
    print("🏃 Running Sentiment Analysis...")
    
    #Apply tonkenize function into Dataframe
    df['Text Content Tokenized'] = df['Text Content'].apply(tokenize_sentences)

    #Apply Sentiment scoring function into Dataframe
    df['Sentiment Score'] = df.apply(lambda x: calculate_sentiment_score(x['Text Content'],afinn_dict), axis=1)

    #Apply normalize function to Dataframe
    df['Normalized Sentiment Score'] = df.apply(lambda x: normalize_score(x['Sentiment Score'],len(x["Text Content"].split())), axis=1)

    #Apply finding extreme sentences function to Dataframe
    df['Extreme Sentences'] = df.apply(lambda x: find_extreme_sentences(x['Text Content Tokenized'],afinn_dict), axis=1)

    #Creating Columns for Most and Least Extreme Sentences with checks
    df['Most Positive Senctence'] = df['Extreme Sentences'].apply(
        lambda x: x[0]['sentence'] if x and isinstance(x, tuple) and x[0] is not None else None)
    df['Most Positive Senctence Score'] = df['Extreme Sentences'].apply(
        lambda x: x[0]['score'] if x and isinstance(x, tuple) and x[0] is not None else None)
    df['Most Negative Senctence'] = df['Extreme Sentences'].apply(
        lambda x: x[1]['sentence'] if x and isinstance(x, tuple) and x[1] is not None else None)
    df['Most Negative Senctence Score'] = df['Extreme Sentences'].apply(
        lambda x: x[1]['score'] if x and isinstance(x, tuple) and x[1] is not None else None)

    #Applying Sliding Window to Dataframe
    df['Sliding Window Results'] = df.apply(lambda x: sliding_window_analysis_words(x['Text Content'],afinn_dict, window_size=10), axis=1)

    #Creating Columns for Most Positive and Negative lines of words from sliding window analysis
    df['Most Positive Line'] = df['Sliding Window Results'].apply(lambda x: x[0]['paragraph'])
    df['Most Positive Line Score'] = df['Sliding Window Results'].apply(lambda x: x[0]['score'])
    df['Most Negative Line'] = df['Sliding Window Results'].apply(lambda x: x[1]['paragraph'])
    df['Most Negative Line Score'] = df['Sliding Window Results'].apply(lambda x: x[1]['score'])
    #Output results into csv
    df.to_csv('data/airlines_review_analysis.csv', index=False)
    
    print("✅ Sentiment Analysis Completed. Saved to airlines_review_analysis.csv.")
    return df

def full_pipeline(df):
    '''
    The main execution function that runs the entire sentiment analysis workflow.
    (Note: The body of this function is incomplete but its role is to drive the process.)

    Args:
        df (pd.DataFrame): The raw input DataFrame that needs analysis.
    Returns:
        pd.DataFrame: The fully analyzed DataFrame returned by run_sentiment_analysis.
    '''
    afinn_dict = load_afinn_lexicon("data/AFINN-en-165.txt")
    df = call_dynamic_prog(df)
    df = load_and_clean_data(df)
    df = run_sentiment_analysis(df, afinn_dict)
    df.to_csv("data/airlines_review_analysis.csv", index=False)
    return df

