from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import term_frequences 
import matplotlib.pyplot as plt
# % matplotlib inline

pos_assoc_df = pd.DataFrame(term_frequences.top_pos)
pos_words = {}
for word, so in pos_assoc_df.values:
    pos_words[word] = so

# Create and generate a word cloud image:
wordcloud = WordCloud(width=600, height=400).generate_from_frequencies(frequencies=pos_words)

# Display the generated image:
wordcloud.to_file("first_review.png")

