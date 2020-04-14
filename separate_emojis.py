import re
import emoji
import pandas as pd
from pandas.core.common import flatten

language = "eng" # eng iben hin
df1 = pd.read_csv("data/trac2020/trac2_"+language+"_train.csv")
df2 = pd.read_csv("data/trac2020/trac2_"+language+"_dev.csv")
df3 = pd.read_csv("data/trac2020/trac2_"+language+"_test.csv")

df = pd.concat([df1.filter(['Text']), df2.filter(['Text']), df3.filter(['Text'])], ignore_index=True)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

tmp_df = df['Text'].apply(lambda x: EMOJI_PATTERN.findall(x))
emojis = set(flatten(list(tmp_df)))
emojis = set("".join(emojis))
print(emojis)

# iterate through all characters of an input text and copy them to the output text.
# if the character is an emoji, ensure that there are whitespaces before and after the emoji in the output text.

def separate_emojis(input_text):
    previous_character = ""
    output_text = ""
    next_character = ""
    for idx,character in enumerate(input_text):
        if character in emojis and not idx == 0 and not previous_character == " ":
        #if emoji and no whitespace before and not string begin:
            output_text = output_text + " "
        #add character (no matter whether it's an emoji or not
        output_text = output_text + character
        #update next character:
        if not idx == (len(text)-1):
            next_character = text[idx+1]
        if character in emojis and not idx == (len(text)-1) and not (next_character == " " or next_character in emojis):
        #if emoji and no whitespace after and not string end:
            output_text = output_text + " "
        previous_character = character
    return output

df1['Text'] = df1['Text'].apply(lambda x: separate_emojis(str(x)))
df2['Text'] = df2['Text'].apply(lambda x: separate_emojis(str(x)))
df3['Text'] = df3['Text'].apply(lambda x: separate_emojis(str(x)))

df1.to_csv("data/trac2020/trac2_"+language+"_train_sep_emoji.csv", index=False)
df2.to_csv("data/trac2020/trac2_"+language+"_dev_sep_emoji.csv", index=False)
df3.to_csv("data/trac2020/trac2_"+language+"_test_sep_emoji.csv", index=False)
