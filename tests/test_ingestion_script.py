import os
import sqlite3
import shutil
import time
from docx import Document
from reportlab.pdfgen import canvas
from foundry_local_sdk import Configuration, FoundryLocalManager
from document_ingestion import index_document, remove_document

def create_test_pdf(filepath, text):
    c = canvas.Canvas(filepath)
    c.drawString(100, 750, text)
    c.save()

def create_test_docx(filepath, text):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filepath)

def create_test_txt(filepath, text):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    db_path = os.path.join('data', 'knowledge.db')
    test_txt = "test_doc_a.txt"
    test_pdf = "test_doc_d.pdf"
    test_docx = "test_doc_e.docx"
    
    print("Setting up embedding client...")
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    model_alias = "qwen3-embedding-0.6b"
    model = manager.catalog.get_model(model_alias)
    if not model.is_cached: model.download()
    if not model.is_loaded: model.load()
    client = model.get_embedding_client()
    
    try:
        # Test A: Index new TXT
        print("\n--- Test A: Index new TXT ---")
        create_test_txt(test_txt, "This is a brand new test document for Test A.\n\nIt has two paragraphs.")
        with open(test_txt, 'rb') as f:
            res_a = index_document(test_txt, f.read(), client, db_path)
        print(f"Result A: {res_a['status']} (Expected: indexed)")
        assert res_a['status'] == 'indexed', "Test A Failed"
        
        # Test B: Index same TXT again
        print("\n--- Test B: Index same TXT ---")
        with open(test_txt, 'rb') as f:
            res_b = index_document(test_txt, f.read(), client, db_path)
        print(f"Result B: {res_b['status']} (Expected: skipped)")
        assert res_b['status'] == 'skipped', "Test B Failed"
        
        # Test C: Modify TXT and index again
        print("\n--- Test C: Modify TXT and index again ---")
        create_test_txt(test_txt, "This is a brand new test document for Test A.\n\nNow it is modified for Test C.")
        with open(test_txt, 'rb') as f:
            res_c = index_document(test_txt, f.read(), client, db_path)
        print(f"Result C: {res_c['status']} (Expected: updated)")
        assert res_c['status'] == 'updated', "Test C Failed"
        
        # Test D: Index PDF
        print("\n--- Test D: Index PDF ---")
        create_test_pdf(test_pdf, "This is a test PDF document.")
        with open(test_pdf, 'rb') as f:
            res_d = index_document(test_pdf, f.read(), client, db_path)
        print(f"Result D: {res_d['status']} (Expected: indexed)")
        assert res_d['status'] == 'indexed', "Test D Failed"
        
        # Test E: Index DOCX
        print("\n--- Test E: Index DOCX ---")
        create_test_docx(test_docx, "This is a test DOCX document.")
        with open(test_docx, 'rb') as f:
            res_e = index_document(test_docx, f.read(), client, db_path)
        print(f"Result E: {res_e['status']} (Expected: indexed)")
        assert res_e['status'] == 'indexed', "Test E Failed"
        
        # Test F: Ensure previously indexed documents remain
        print("\n--- Test F: Check previous docs ---")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT source FROM documents')
        docs = [row[0] for row in c.fetchall()]
        conn.close()
        print(f"Current docs: {docs}")
        # Assuming sqlite.txt, rag.txt, python.txt might be there, plus our 3 tests.
        assert test_txt in docs, "Test F Failed: TXT missing"
        assert test_pdf in docs, "Test F Failed: PDF missing"
        assert test_docx in docs, "Test F Failed: DOCX missing"
        print("Result F: Docs exist (Expected: PASS)")
        
        # Test G: Remove test document
        print("\n--- Test G: Remove test document ---")
        remove_document(db_path, test_txt)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT source FROM documents WHERE source=?', (test_txt,))
        exists_doc = c.fetchone()
        c.execute('SELECT count(*) FROM chunks WHERE source=?', (test_txt,))
        num_chunks = c.fetchone()[0]
        conn.close()
        print(f"Document entry exists: {exists_doc is not None}")
        print(f"Chunks remaining: {num_chunks}")
        assert exists_doc is None, "Test G Failed: Document metadata not removed"
        assert num_chunks == 0, "Test G Failed: Chunks not removed"
        
        print("\nALL TESTS PASSED!")
        
    finally:
        model.unload()
        # Clean up database
        remove_document(db_path, test_txt)
        remove_document(db_path, test_pdf)
        remove_document(db_path, test_docx)
        
        # Clean up files
        for f in [test_txt, test_pdf, test_docx]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()
