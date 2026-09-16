from search import search_prompt


def main():
    print("Chat RAG iniciado! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando chat. Até logo!")
            break

        resposta = search_prompt(pergunta)
        print(f"RESPOSTA: {resposta}\n")


if __name__ == "__main__":
    main()
