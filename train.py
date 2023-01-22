import os
import tiktoken

with open('input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

print("char length", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

enc = tiktoken.get_encoding('gpt2')

encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

print(encode("hii there"))
print(decode(encode("hii there")))
