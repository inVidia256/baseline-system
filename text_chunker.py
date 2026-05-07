import re
from typing import List

def chunk_text(text, chunk_size=8192, overlap=800, truncate_to=None, language="english"):
    if not text:
        return []
    
    sentences = re.split(r'(?<=[。！？?!\.])(?![a-zA-Z]\w)|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 1 and len(text) > 100:
        sentences = re.split(r'(?<=[。！？?!\.])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
            
        sent_length = len(sent)
        if sent_length > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
            
            long_chunks = _split_long_sentence(sent, chunk_size, overlap)
            chunks.extend(long_chunks[:-1]) 
            if long_chunks:
                current_chunk = long_chunks[-1]
                current_length = len(current_chunk)
            continue
        
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            if overlap > 0:
                overlap_text = _get_overlap_at_sentence_boundary(
                    current_chunk, overlap, sentences[:i]
                )
                current_chunk = overlap_text
                current_length = len(overlap_text)
            else:
                current_chunk = ""
                current_length = 0
        
        if current_chunk:
            if re.search(r'[a-zA-Z]', sent[-1:] if sent else '') and re.search(r'[a-zA-Z]', current_chunk[-1:] if current_chunk else ''):
                current_chunk += " " + sent
                current_length += 1 + sent_length
            else:
                current_chunk += sent
                current_length += sent_length
        else:
            current_chunk = sent
            current_length = sent_length
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    if truncate_to is not None and truncate_to > 0:
        chunks = [_truncate_at_sentence(chunk, truncate_to) for chunk in chunks]
    
    return chunks

def _split_long_sentence(sentence: str, max_len: int, overlap: int) -> List[str]:
    chunks = []
    words = re.findall(r'\b\w+\b|[^\w\s]', sentence)
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word)
        
        if current_length + word_len + (1 if current_chunk else 0) > max_len and current_chunk:
            chunk_text = ''.join(current_chunk)  
            chunks.append(chunk_text)
            if overlap > 0:
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) <= overlap:
                        overlap_words.insert(0, w)
                        overlap_length += len(w)
                    else:
                        break
                current_chunk = overlap_words
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(word)
        current_length += len(word)
    
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def _get_overlap_at_sentence_boundary(chunk: str, target_overlap: int, 
                                     previous_sentences: List[str]) -> str:
   
    if target_overlap <= 0:
        return ""
    
    overlap_sentences = []
    overlap_length = 0
    
    for sent in reversed(previous_sentences):
        sent = sent.strip()
        if not sent:
            continue
            
        sent_len = len(sent)
        if overlap_length + sent_len > target_overlap and overlap_sentences:
            break
            
        overlap_sentences.insert(0, sent)
        overlap_length += sent_len
    
    return ''.join(overlap_sentences)  

def _truncate_at_sentence(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    
    truncated = text[:max_len]
    
    last_sentence_end = -1
    for i in range(max_len - 1, max(0, max_len - 100), -1):
        if truncated[i] in '.!?。！？':
            if i > 1 and truncated[i-1].isalpha() and truncated[i] == '.':
                continue
            last_sentence_end = i
            break
    
    if last_sentence_end > 0:
        return truncated[:last_sentence_end + 1]
    
    last_newline = truncated.rfind('\n\n')
    if last_newline > 0:
        return truncated[:last_newline]
    
    for punct in ['，', '；', ',', ';']:
        last_punct = truncated.rfind(punct)
        if last_punct > 0:
            return truncated[:last_punct + 1]
    
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space]
    
    return truncated