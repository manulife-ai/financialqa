import re

def cleanup_whitespace(s:str):
    return re.sub("\s+", " ", s)

def cleanup_multiple_underlines(s:str):
    return re.sub(r"_{2,}", " ", s)

def cleanup_escapechar(s:str):
    return re.sub("\\'", "", s)

def cleanup_copyrights_characters(s:str):
    return re.sub('Copyright ©.+(Manulife)', '', s)

def preprocess_text(text):
    text = cleanup_whitespace(text)
    text = cleanup_multiple_underlines(text)
    text = cleanup_copyrights_characters(text)
    text = cleanup_escapechar(text)
    return text