from bahdanaunmt.utils import train

encoder, decoder = train('eng', 'fin', './data/eng-fin.tsv', n_epochs=5)
