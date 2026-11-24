class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        # self.vocab=[]

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        # ["hello world", "this is a test", "hello test"]
        self.word_to_id["<PAD>"]=0
        self.word_to_id["<UNK>"]=1
        self.word_to_id["<BOS>"]=2
        self.word_to_id["<EOS>"]=3

        # text=(" ".join(texts)).lower()
        # partition=sorted(set((list(text.split(" ")))))
        #"hello world", "this is a test" and "hello test"
        # vocab_content= ["hello world", "this is a test", "hello test"]
        v=(" ".join(texts)).lower()
        vocab=list(sorted(set((list(v.split(" "))))))
        self.vocab_size=len(vocab)+4
        
        for i in range(4,self.vocab_size):
            self.word_to_id[vocab[i-4]]=i
        # arr=[]
        # for token in texts:
        #     arr.append(self.word_to_id.get(token,1))
        # return arr

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        txt=text.lower()
        if txt=="":
            return []
        return [self.word_to_id.get(token,1) for token in txt.split(" ")]
        
         
        

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        self.id_to_words={v:k for k,v in self.word_to_id.items()}
        lst=[]
        for id in ids:
            if id in self.id_to_words:
                lst.append(self.id_to_words[id])
            else:
                lst.append("<UNK>")

        return " ".join(lst)
                