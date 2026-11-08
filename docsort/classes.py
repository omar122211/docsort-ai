"""The document types DocSort recognises, and where their training images come from."""

# label → (dataset, original label id). RVL-CDIP ids: 2 email, 4 form, 6 invoice, 7 letter, 12 resume, 13 scientific publication
CLASSES = ["email", "form", "invoice", "letter", "receipt", "resume", "scientific_publication"]
RVL_IDS = {2: "email", 4: "form", 6: "invoice", 7: "letter", 12: "resume", 13: "scientific_publication"}

LABELS = {
    "email": "Email",
    "form": "Form",
    "invoice": "Invoice",
    "letter": "Letter",
    "receipt": "Receipt",
    "resume": "Resume / CV",
    "scientific_publication": "Scientific paper",
}
