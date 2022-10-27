from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

b = bleu(
    ['The candidate has no alignment to any of the references'.split()],
    'John loves Mary'.split(),
    (1,),
)

print(b)

a = sentence_bleu(
    ['It is a place of quiet contemplation .'.split()],
    'It is .'.split(),
    smoothing_function=SmoothingFunction().method4,
)*100


print(a)
