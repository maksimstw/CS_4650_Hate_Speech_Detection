import pandas as pd


LABELS = {
    'noHate': 0,
    'hate': 1
}

def get_text(file_id):
    with open(f'hate-speech-dataset/all_files/{file_id}.txt') as f:
        return f.readline()

if __name__ == '__main__':
    df = pd.read_csv('hate-speech-dataset/annotations_metadata.csv')
    df.index.name = 'doc_id'  
    df['text'] = df['file_id'].apply(get_text)
    df['label'].replace(LABELS, inplace=True)
    df.rename(columns={'label': 'is_hate'}, inplace=True)
    df.to_csv('stormfront.tsv', sep='\t', columns = ['text', 'is_hate'])