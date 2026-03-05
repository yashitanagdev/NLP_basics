from nltk.stem import PorterStemmer, RegexpStemmer, SnowballStemmer

stemmer = PorterStemmer()

words = ["running", "jumps", "fairly", "faster"]
for word in words:
    print(stemmer.stem(word))

print(stemmer.stem('congratulations '))

reg_exp_stemmer = RegexpStemmer('ing$|s$|en$', min = 4)
print(reg_exp_stemmer.stem('eaten'))

snowball_stemmer = SnowballStemmer("english")
print(snowball_stemmer.stem('fairly'))