from utils import WordVocabulary

train_path = 'input/train.txt'
dev_path = 'typo_data/typo_dev.txt'
test_path = 'typo_data/typo_test.txt'
out_path = 'typo_data/typo_vocab.txt'

word_vocab = WordVocabulary(train_path, dev_path, test_path, True)
with open(out_path, 'w', encoding='utf8')as f:
    f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))
