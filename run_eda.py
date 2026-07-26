import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Create output directory for plots
os.makedirs('eda_outputs', exist_ok=True)

print("Loading Train.csv...")
df = pd.read_csv('Train.csv', on_bad_lines='skip')

print(f"Total records: {len(df)}")
print("\n--- Language Distribution ---")
lang_counts = df['language'].value_counts()
print(lang_counts)

# Plot Language Distribution
plt.figure(figsize=(10, 6))
lang_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Samples per Language', fontsize=14)
plt.xlabel('Language Code', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/language_distribution.png')
plt.close()

# Transcription analysis
print("\n--- Transcription Length Analysis ---")
df['char_length'] = df['transcription'].apply(lambda x: len(str(x)))
df['word_length'] = df['transcription'].apply(lambda x: len(str(x).split()))

print("Character length stats:")
print(df['char_length'].describe())
print("\nWord length stats:")
print(df['word_length'].describe())

# Plot Length Distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
df['char_length'].plot(kind='hist', bins=50, ax=axes[0], color='lightgreen', edgecolor='black')
axes[0].set_title('Character Length Distribution')
axes[0].set_xlabel('Number of Characters')

df['word_length'].plot(kind='hist', bins=50, ax=axes[1], color='salmon', edgecolor='black')
axes[1].set_title('Word Length Distribution')
axes[1].set_xlabel('Number of Words')
plt.tight_layout()
plt.savefig('eda_outputs/length_distributions.png')
plt.close()

# Vocabulary extraction (to check for weird characters/numbers)
print("\n--- Vocabulary Analysis ---")
all_text = " ".join(df['transcription'].astype(str).tolist()).lower()
unique_chars = sorted(list(set(all_text)))
with open('eda_outputs/vocabulary.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total unique characters: {len(unique_chars)}\n")
    f.write(f"Characters: {''.join(unique_chars)}\n")
    
    # Find if there are numbers
    numbers_found = re.findall(r'\d', all_text)
    if numbers_found:
        f.write(f"Found numbers in transcriptions: {set(numbers_found)}\n")
    else:
        f.write("No numbers found in transcriptions.\n")

print("Vocabulary written to eda_outputs/vocabulary.txt")

# Test set analysis
print("\n--- Loading Test.csv ---")
test_df = pd.read_csv('Test.csv')
print(f"Total test records: {len(test_df)}")
# Extract language from test ID (e.g., 'lug_96114' -> 'lug')
test_df['inferred_language'] = test_df['ID'].apply(lambda x: x.split('_')[0])
test_lang_counts = test_df['inferred_language'].value_counts()
print("\nInferred Language Distribution in Test Set:")
print(test_lang_counts)

# Plot Test vs Train languages
plt.figure(figsize=(10, 6))
test_lang_counts.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Inferred Distribution of Samples per Language (Test Set)', fontsize=14)
plt.xlabel('Language Code', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/test_language_distribution.png')
plt.close()

print("\nEDA completed. Plots saved to 'eda_outputs' directory.")
