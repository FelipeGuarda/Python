import csv
from collections import Counter

with open(r'C:\Users\USUARIO\Dev\Python\Claude\species-classifier\old animal data DB.csv', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))

# Inspect Animal column values
animal_vals = Counter(r.get('Animal', '').strip() for r in rows)
print("Animal column values:", dict(animal_vals.most_common(10)))

# Inspect Especie column values (sample)
especie_vals = Counter(r.get('Especie', '').strip() for r in rows)
print("\nEspecie column values (top 30):")
print(f"{'Species':<30} {'Count':>6}  {'%':>5}")
print('-' * 45)
total = sum(n for sp, n in especie_vals.items() if sp)
for sp, n in especie_vals.most_common(30):
    if sp:
        pct = 100 * n / total
        print(f"{sp:<30} {n:>6}  {pct:>4.1f}%")
