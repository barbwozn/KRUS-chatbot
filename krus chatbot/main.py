from law_data import ask
from tabular_data import answer

print("AgroBot – wpisz pytanie (lub 'exit' aby wyjść)\n")

while True:
    try:
        user_msg = input("Ty: ").strip()
    except EOFError:
        break
    if not user_msg:
        continue
    if user_msg.lower() in {"exit","quit","q"}:
        print("koniec")
        break

    ust = ask(user_msg)  
    data = answer(user_msg)
    print("KRUS:", "\n", "USTAWA", "\n", ust, "\n", "DANE TABELARYCZNE", "\n", data, "\n")