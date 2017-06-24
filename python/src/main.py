from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

def main():
    lines = read_lines()

    vectorizer = CountVectorizer(binary='true')

    train_documents = vectorizer.fit_transform(lines[0])
    train_labels = lines[1]

    #Training phase
    classifier = BernoulliNB().fit(train_documents, train_labels)

    #Test phase
    classifier.predict(vectorizer.transform(['this sucks!']))

def read_lines():
    lines = []

    with open('./data/sentiments/amazon_cells_labelled.txt', 'r') as text_file:
        lines += text_file.read().split('\n')

    with open('./data/sentiments/imdb_labelled.txt', 'r') as text_file:
        lines += text_file.read().split('\n')

    with open('./data/sentiments/yelp_labelled.txt') as text_file:
        lines += text_file.read().split('\n')

    lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t')[1] != '']

    train_documents = [line[0] for line in lines]
    train_labels = [int(line[1]) for line in lines]

    return (train_documents, train_labels)

main()
