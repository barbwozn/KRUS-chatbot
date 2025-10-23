from law_data import ask
from tabular_data import answer

print("AgroBot – wpisz pytanie (lub 'exit' aby wyjść)\n")

key_words = ["dane"]
used_module = None 

while True:
    try:
        user_msg = input("Ty: ").strip()
    except EOFError:
        break

    if not user_msg:
        print("(pusta wiadomość)\n")
        continue

    if user_msg.lower() in {"exit", "quit", "q"}:
        print("koniec")
        break

    if any(kw in user_msg.lower() for kw in key_words):
        used_module = answer(user_msg)
    else:
        used_module = ask(user_msg)

    print(used_module, "\n")