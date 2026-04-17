with open('src/report_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '\u2014': '-',   # em dash —
    '\u2013': '-',   # en dash –
    '\u2018': "'",   # left single quote '
    '\u2019': "'",   # right single quote '
    '\u201c': '"',   # left double quote "
    '\u201d': '"',   # right double quote "
    '\u2026': '...',  # ellipsis …
    '\u2022': '-',   # bullet •
}

count = 0
for char, replacement in replacements.items():
    occurrences = content.count(char)
    if occurrences:
        print(f"Replacing {occurrences}x U+{ord(char):04X} ({char}) -> '{replacement}'")
        content = content.replace(char, replacement)
        count += occurrences

with open('src/report_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal replacements: {count}")
print("Done.")