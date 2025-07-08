from typing import Iterable
import pandas as pd
from langchain_community.vectorstores import FAISS
import streamlit as st

from .pipeline import RAGPipeline

TEMPLATE_FIELDS = [
    "Ente erogatore",
    "Titolo dell'avviso",
    "Descrizione aggiuntiva",
    "Beneficiari",
    "Apertura",
    "Chiusura",
    "Dotazione finanziaria",
    "Contributo",
    "Note",
    "Link",
    "Key Words",
    "Aperto (si/no)",
]

TEMPLATE_QUESTIONS = {
    "Ente erogatore": (
        "Scrivi solo il nome esatto dell’ente erogatore di questo bando, "
        "scegliendolo dalle prime tre pagine. Se non lo trovi, deduci quello più "
        "probabile dal testo. Solo il nome, nessuna spiegazione."
    ),
    "Titolo dell'avviso": (
        "Scrivi solo il titolo ufficiale dell’avviso, così come appare o come "
        "puoi dedurlo dalle prime tre pagine. Solo la dicitura, nessuna spiegazione."
    ),
    "Descrizione aggiuntiva": (
        "Scrivi una sola frase molto breve (massimo 25 parole) che riassume l’intero bando. "
        "Solo la frase, senza spiegazioni."
    ),
    "Beneficiari": (
        "Scrivi solo i beneficiari principali di questo bando, anche dedotti dal testo. "
        "Solo l’elenco, senza spiegazioni."
    ),
    "Apertura": (
        "Scrivi solo la data di apertura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata."
    ),
    "Chiusura": (
        "Scrivi solo la data di chiusura (formato GG/MM/AAAA), anche dedotta dal testo se non è esplicitata."
    ),
    "Dotazione finanziaria": (
        "Qual è la dotazione finanziaria totale del bando? Scrivi solo la cifra o il valore principale "
        "della dotazione finanziaria, anche se devi dedurlo dal testo."
    ),
    "Contributo": (
        "Qual è il contributo previsto per i beneficiari? Scrivi solo la cifra o percentuale "
        "principale del contributo previsto, anche se la deduci dal testo."
    ),
    "Note": (
        "Scrivi solo una nota rilevante, anche se la deduci dal testo. Solo la nota, senza spiegazioni."
    ),
    "Link": (
        "Scrivi solo il link (URL) principale trovato nel testo, oppure deducilo se presente in altro modo."
    ),
    "Key Words": (
        "Scrivi solo tre parole chiave, anche dedotte dal testo, separate da virgola e senza spiegazioni."
    ),
    "Aperto (si/no)": (
        "Rispondi solo con 'si' o 'no' se il bando è ancora aperto; deduci la risposta dal testo e dalle date."
    ),
}


def fill_template_for_all_files(pipeline: RAGPipeline, top_k: int = 3) -> pd.DataFrame:
    file_to_docs = {}
    for doc in pipeline.documents:
        file = doc.metadata["file_name"]
        file_to_docs.setdefault(file, []).append(doc)

    rows = []
    progress = st.progress(0, text="Estrazione in corso...")
    files = list(file_to_docs.keys())
    total_steps = len(files) * len(TEMPLATE_FIELDS)
    step = 0

    for file in files:
        row = {}
        docs_this_file = file_to_docs[file]
        first_3_chunks = docs_this_file[:3]

        for field in TEMPLATE_FIELDS:
            question = TEMPLATE_QUESTIONS[field]
            retriever = FAISS.from_documents(
                first_3_chunks if field in ["Ente erogatore", "Titolo dell'avviso"] else docs_this_file,
                pipeline.embedding_wrapper,
            ).as_retriever()
            relevant_docs = retriever.get_relevant_documents(question, k=top_k)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            if field == "Descrizione aggiuntiva":
                question = (
                    "Scrivi una sola frase molto breve (massimo 25 parole) che riassume il bando."
                )

            prompt = f"""
Testo selezionato del bando (estratto tramite ricerca semantica):
{context}

Domanda: {question}
Rispondi solo sulla base del testo fornito sopra. Non aggiungere spiegazioni.
"""

            answer = pipeline.chat_model.ask_about_document(context, prompt)
            row[field] = answer.strip()

            st.markdown(
                f"**[{file}]** - Campo: `{field}`\n- Prompt:\n```{prompt[:500]}...```\n- Risposta GPT:\n```{answer[:300]}...```\n---"
            )
            step += 1
            progress.progress(step / total_steps, text=f"Completato {step}/{total_steps}")

        row["File"] = file
        rows.append(row)

    progress.empty()
    return pd.DataFrame(rows)
