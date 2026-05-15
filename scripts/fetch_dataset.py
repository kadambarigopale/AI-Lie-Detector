import urllib.request
import pandas as pd
import sys

# Common github repositories hosting the deceptive opinion spam dataset
urls = [
    "https://raw.githubusercontent.com/swcwang/deceptive_hotel_review/master/data/deceptive-opinion.csv",
    "https://raw.githubusercontent.com/Gargi-Pawar/Deceptive-Opinion-Spam-Detection/master/deceptive-opinion.csv",
    "https://raw.githubusercontent.com/muntasirhsn/Deceptive-Opinion-Spam-Corpus/master/deceptive-opinion.csv"
]
output_file = "dataset.csv"

success = False
for url in urls:
    print(f"Attempting to download from {url}...")
    try:
        urllib.request.urlretrieve(url, output_file)
        
        # Verify and format it
        df = pd.read_csv(output_file)
        
        # We need columns 'text' and 'label' (with values 'truthful' / 'deceptive')
        if 'deceptive' in df.columns:
            df.rename(columns={'deceptive': 'label'}, inplace=True)
        
        print("Found columns:", df.columns.tolist())
        # The text column is usually 'text'
        
        df.to_csv(output_file, index=False)
        print(f"Success! Saved formatted dataset with {len(df)} rows to {output_file}")
        success = True
        break
    except Exception as e:
        print(f"Failed: {e}")

if not success:
    print("ERROR: Could not download the dataset from any source.")
    sys.exit(1)
