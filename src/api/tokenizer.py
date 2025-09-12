import json
import re
from typing import List, Dict, Optional, Tuple, Union
from transformers import PreTrainedTokenizer


def load_vocab(file_path="bnc_word2c5.json", top_k=40000):
    vocab = json.loads(open(file_path).read())
    processed_vocab = {}

    for word, data in vocab.items():
        total_cnt = sum(list(data.values()))
        pos_info = {k: v / total_cnt for k, v in data.items()}
        processed_vocab[word] = {"total": total_cnt, "pos_info": pos_info}

    # Sort by total count and keep only top_k entries
    processed_vocab = sorted(processed_vocab.items(), key=lambda x: x[1]["total"], reverse=True)
    processed_vocab = processed_vocab[:top_k]
        
    return dict(processed_vocab)


class BNCTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer for BNC vocabulary with naive tokenization
    """
    
    vocab_files_names = {"vocab_file": "bnc_word2c5.json"}
    
    def __init__(
        self,
        vocab_file="bnc_word2c5.json",
        top_k=40000,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        clean_text=True,
        lowercase=True,
        **kwargs
    ):
        # Load BNC vocabulary
        try:
            self.bnc_vocab = load_vocab(vocab_file, top_k=top_k)
        except FileNotFoundError:
            print(f"Warning: {vocab_file} not found. Using empty vocabulary.")
            self.bnc_vocab = {}
        
        # Create token to ID mapping
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens first
        special_tokens = [unk_token, bos_token, eos_token]
        current_id = 0
        
        for token in special_tokens:
            if token is not None:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        # Add vocabulary tokens
        for word in sorted(self.bnc_vocab.keys()):
            if word not in self.token_to_id:
                self.token_to_id[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        self.lowercase = lowercase
        self.clean_text = clean_text
        
        super().__init__(
            unk_token=unk_token,
            pad_token=eos_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            lowercase=lowercase,
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        """Return the size of vocabulary"""
        return len(self.token_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary"""
        return self.token_to_id.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Naive tokenization with proper handling of contractions
        """
        if self.lowercase:
            text = text.lower()
        
        if self.clean_text:
            text = self._clean_text(text)
        
        # Updated regex to handle contractions and common patterns
        # This pattern matches:
        # - Words with optional contractions (word + 's, 't, 're, etc.)
        # - Individual punctuation marks
        tokens = re.findall(r"\w+(?:'[a-z]+)?|[^\w\s]", text)
        
        # Convert unknown words to UNK token, but first try to handle contractions
        result_tokens = []
        for token in tokens:
            if token in self.bnc_vocab:
                result_tokens.append(token)
            else:
                # If token contains apostrophe, try to split and check parts
                if "'" in token:
                    handled = self._handle_contraction(token)
                    result_tokens.extend(handled)
                else:
                    result_tokens.append(self.unk_token)
        
        return result_tokens
    
    def _handle_contraction(self, token: str) -> List[str]:
        """
        Handle contractions by checking if parts exist in vocabulary
        """
        # Split on apostrophe
        parts = token.split("'")
        if len(parts) == 2:
            base_word = parts[0]
            contraction = "'" + parts[1]
            
            result = []
            
            # Check if base word is in vocab
            if base_word in self.bnc_vocab:
                result.append(base_word)
            else:
                result.append(self.unk_token)
            
            # Check if contraction is in vocab
            if contraction in self.bnc_vocab:
                result.append(contraction)
            else:
                result.append(self.unk_token)
            
            return result
        else:
            # Complex case with multiple apostrophes, just use UNK
            return [self.unk_token]
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        return self.id_to_token.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string with proper contraction handling"""
        if not tokens:
            return ""
        
        result = ""
        for i, token in enumerate(tokens):
            # Skip special tokens in reconstruction
            if token in [self.pad_token, self.cls_token, self.sep_token, 
                        self.mask_token, self.bos_token, self.eos_token]:
                continue
            
            # Add the token
            result += token
            
            # Add space after token unless:
            # 1. It's the last token
            # 2. The current token is punctuation
            # 3. The next token is punctuation or starts with apostrophe
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                
                # Don't add space if next token starts with apostrophe (contraction)
                if next_token.startswith("'"):
                    continue
                
                # Don't add space before/after punctuation
                if (not re.match(r'^[^\w\s]$', token) and 
                    not re.match(r'^[^\w\s]$', next_token) and
                    next_token not in [self.pad_token, self.bos_token, self.eos_token]):
                    result += " "
        
        return result.strip()
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Create mask for special tokens
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to file"""
        import os
        
        if filename_prefix is None:
            filename_prefix = ""
        
        vocab_file = os.path.join(save_directory, filename_prefix + "bnc_word2c5.json")
        
        # Save the original BNC vocabulary format
        original_vocab = {}
        for word, data in self.bnc_vocab.items():
            # Reconstruct original format from processed data
            total = data["total"]
            pos_info = data["pos_info"]
            word_data = {pos: int(prob * total) for pos, prob in pos_info.items()}
            original_vocab[word] = word_data
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(original_vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)
    
    def get_pos_info(self, word: str) -> Optional[Dict[str, float]]:
        """Get POS information for a word"""
        word_lower = word.lower() if self.lowercase else word
        if word_lower in self.bnc_vocab:
            return self.bnc_vocab[word_lower]["pos_info"]
        return None
    
    def get_most_likely_pos(self, word: str) -> Optional[str]:
        """Get the most likely POS tag for a word"""
        pos_info = self.get_pos_info(word)
        if pos_info:
            return max(pos_info.items(), key=lambda x: x[1])[0]
        return None