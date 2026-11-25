import tempfile
import unittest
import warnings
from pathlib import Path

import retriever
from pypdf import PdfWriter


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.knowledge_dir = Path(self.temp_dir.name) / "knowledge"
        self.knowledge_dir.mkdir()
        self.original_knowledge_dir = retriever.KNOWLEDGE_DIR
        retriever.KNOWLEDGE_DIR = self.knowledge_dir
        retriever._load_chunks.cache_clear()
        self.addCleanup(self._restore_retriever_state)

    def _restore_retriever_state(self) -> None:
        retriever._load_chunks.cache_clear()
        retriever.KNOWLEDGE_DIR = self.original_knowledge_dir

    def test_retrieves_relevant_source(self) -> None:
        (self.knowledge_dir / "company_policy.txt").write_text(
            "Employees request annual leave from their manager for approval.",
            encoding="utf-8",
        )
        (self.knowledge_dir / "python.txt").write_text(
            "Python functions accept arguments and return values.",
            encoding="utf-8",
        )

        results = retriever.retrieve_documents("annual leave approval")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "company_policy.txt")
        self.assertIn(
            "[Source: company_policy.txt]",
            retriever.search_knowledge("annual leave"),
        )

    def test_stop_word_only_query_has_no_results(self) -> None:
        (self.knowledge_dir / "article.txt").write_text(
            "A short article about retrieval.",
            encoding="utf-8",
        )

        self.assertEqual(retriever.retrieve_documents("what is the"), [])

    def test_retrieves_document_from_underscored_filename(self) -> None:
        (self.knowledge_dir / "company_policy.txt").write_text(
            "The policy covers publishing, printing quality, confidentiality, "
            "and workplace safety.",
            encoding="utf-8",
        )

        results = retriever.retrieve_documents(
            "what are the main points in company_policy"
        )

        self.assertTrue(results)
        self.assertEqual(results[0].source, "company_policy.txt")
        self.assertIn("workplace safety", results[0].text)

    def test_reads_named_knowledge_document(self) -> None:
        content = "Company policy source text."
        path = self.knowledge_dir / "company_policy.txt"
        path.write_text(content, encoding="utf-8")

        self.assertEqual(
            retriever.read_knowledge_document("company_policy.txt"),
            content,
        )
        self.assertEqual(
            retriever.read_knowledge_document(str(path)),
            content,
        )

    def test_knowledge_reader_rejects_paths_outside_knowledge(self) -> None:
        outside_file = self.knowledge_dir.parent / "outside.txt"
        outside_file.write_text("Outside knowledge directory.", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "inside knowledge"):
            retriever.read_knowledge_document("../outside.txt")

    def test_text_content_with_pdf_extension_is_loaded_without_warning(self) -> None:
        (self.knowledge_dir / "article.pdf").write_text(
            "A text article stored with a PDF extension.",
            encoding="utf-8",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            documents = retriever.load_documents()

        self.assertIn("article.pdf", documents)
        self.assertEqual(caught, [])

    def test_pdf_without_extractable_text_fails_explicitly(self) -> None:
        path = self.knowledge_dir / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with path.open("wb") as file:
            writer.write(file)

        with self.assertRaisesRegex(RuntimeError, "No extractable text"):
            retriever._read_pdf(path)


if __name__ == "__main__":
    unittest.main()
