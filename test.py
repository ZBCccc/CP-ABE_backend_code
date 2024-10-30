import hashlib
def str_to_int_unicode_mod_p(text, p):
    # 将字符串中每个字符转换为Unicode码点值
    return sum(ord(char) for char in text) % p

# 示例
print(str_to_int_unicode_mod_p("b", 66666))


def str_to_int_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)

print(str_to_int_hash("你好"))
