import pandas as pd
import random
from Crypto.Cipher import AES, DES, DES3, Blowfish, ARC4, ChaCha20, CAST
from Crypto.Random import get_random_bytes

def generate_random_plaintext(min_length=8, max_length=32):
    length = random.randint(min_length, max_length)
    return get_random_bytes(length)

def format_ciphertext(ciphertext):
    return ' '.join(ciphertext[i:i + 2].upper() for i in range(0, len(ciphertext), 2))

def encrypt_aes(plaintext):
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'AES', key.hex()

def encrypt_3des(plaintext):
    key = DES3.adjust_key_parity(get_random_bytes(24))  # 24 bytes key for 3DES
    cipher = DES3.new(key, DES3.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), '3DES', key.hex()

def encrypt_cast5(plaintext):
    key = get_random_bytes(16)
    cipher = CAST.new(key, CAST.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'CAST5', key.hex()

# Existing symmetric algorithms
def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)
    cipher = Blowfish.new(key, Blowfish.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return format_ciphertext((cipher.nonce + tag + ciphertext).hex()), 'Blowfish', key.hex()

def encrypt_chacha20(plaintext):
    key = get_random_bytes(32)
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext((cipher.nonce + ciphertext).hex()), 'ChaCha20', key.hex()

def encrypt_rc4(plaintext):
    key = get_random_bytes(16)
    cipher = ARC4.new(key)
    ciphertext = cipher.encrypt(plaintext)
    return format_ciphertext(ciphertext.hex()), 'RC4', key.hex()

def create_symmetric_dataset(num_samples=1000):
    data_with_key = []
    data_without_key = []
    
    for _ in range(num_samples):
        plaintext = generate_random_plaintext()
        
        for encrypt_func in [encrypt_aes, encrypt_3des, encrypt_cast5, encrypt_blowfish, encrypt_chacha20, encrypt_rc4]:
            ciphertext, algorithm, key = encrypt_func(plaintext)
            data_with_key.append([algorithm, plaintext.hex(), ciphertext, key])
            data_without_key.append([algorithm, plaintext.hex(), ciphertext])
    
    df_with_key = pd.DataFrame(data_with_key, columns=['Algorithm', 'Plaintext', 'Ciphertext', 'Key'])
    df_without_key = pd.DataFrame(data_without_key, columns=['Algorithm', 'Plaintext', 'Ciphertext'])
    
    df_with_key.to_csv('symmetric_with_key.csv', index=False)
    df_without_key.to_csv('symmetric_without_key.csv', index=False)

create_symmetric_dataset(num_samples=100000)
