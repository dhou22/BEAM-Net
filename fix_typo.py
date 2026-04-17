with open('src/report_generator.py', 'r') as f:
    content = f.read()

before = len(content)
content = content.replace(',ew_x=', ', new_x=')
after = len(content)

with open('src/report_generator.py', 'w') as f:
    f.write(content)

print(f"Fixed {before - after} chars" if before != after else "Replacement made (same length)")
print("Done")