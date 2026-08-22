from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

class AEOWUNTokenizer:
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, files):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            show_progress=True
        )
        self.tokenizer.train(files, trainer)

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    t = AEOWUNTokenizer(vocab_size=8192)
    corpus_path = "data/corpus.txt"
    if os.path.exists(corpus_path):
        print(f"Training tokenizer on {corpus_path}...")
        t.train([corpus_path])
        t.save("data/aeowun_tokenizer.json")
        print("Tokenizer saved to data/aeowun_tokenizer.json")
        
        # Test round-trip
        test_text = "def calculate_loss(x, y): return None [AEOWUN_ERROR]: OUT_OF_DOMAIN"
        encoded = t.encode(test_text)
        decoded = t.decode(encoded)
        print(f"Test Encode: {encoded[:10]}...")
        print(f"Test Decode: {decoded}")
        assert decoded.strip() == test_text.strip(), f"Round-trip failed: '{decoded}' != '{test_text}'"
        print("Round-trip test passed.")
    else:
        print(f"Corpus file not found at {corpus_path}")
