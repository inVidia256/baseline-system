# evaluator.py
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

def evaluate_summary(reference, candidate):
    reference = str(reference).strip()
    candidate = str(candidate).strip()
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
    except Exception as e:
        print(f"ROUGE计算错误: {e}")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bleu': 0.0
        }
    
    try:
        ref_tokens = nltk.word_tokenize(reference.lower())
        can_tokens = nltk.word_tokenize(candidate.lower())
        
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([ref_tokens], can_tokens, 
                                  smoothing_function=smoothing)
    except Exception as e:
        print(f"BLEU计算错误: {e}")
        bleu_score = 0.0
    
    return {
        'rouge1': scores['rouge-1']['f'],
        'rouge2': scores['rouge-2']['f'],
        'rougeL': scores['rouge-l']['f'],
        'bleu': bleu_score
    }