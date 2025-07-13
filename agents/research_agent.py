import arxiv
import fitz  # PyMuPDF
import os
import tempfile
import requests
from typing import List, Dict
from retrievers.ensemble_retriever import EnsembleRetriever
from utils.logger import api_logger
from utils.error_handler import safe_execute, ErrorContext
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ResearchAgent:
    def __init__(self, max_results: int = 10, chunk_size: int = 500, chunk_overlap: int = 100):
        self.retriever = EnsembleRetriever()
        self.max_results = max_results
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        api_logger.info(f"ResearchAgent initialized with max_results={max_results}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def search(self, query: str) -> List[Dict]:
        """Search arXiv for relevant research papers."""
        try:
            api_logger.log_agent_activity("ResearchAgent", "search", query=query)
            
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            for result in search.results():
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "published": result.published.isoformat(),
                    "categories": result.categories
                })
            
            api_logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e, 
                f"Error searching arXiv for query: {query}",
                "Check internet connection and arXiv availability"
            )
            return []

    def download_pdf(self, url: str) -> str:
        """Download a PDF to a temporary file and return its path."""
        try:
            api_logger.debug(f"Downloading PDF from: {url}")
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: Failed to download PDF from {url}")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(response.content)
            temp_file.close()
            
            file_size = len(response.content)
            api_logger.debug(f"PDF downloaded successfully. Size: {file_size} bytes")
            return temp_file.name
            
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout downloading PDF from {url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading PDF: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error downloading PDF: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF using PyMuPDF."""
        try:
            api_logger.debug(f"Extracting text from PDF: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                raise Exception(f"PDF file not found: {pdf_path}")
            
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(doc):
                try:
                    text = page.get_text()
                    full_text += text + "\n"
                except Exception as e:
                    api_logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            doc.close()
            
            # Clean up the text
            full_text = full_text.strip()
            
            api_logger.debug(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                f"Error extracting text from PDF: {pdf_path}",
                "Check if PDF is corrupted or password protected"
            )
            return ""

    def chunk_text_with_langchain(self, text: str) -> List[str]:
        """Chunk text using LangChain's RecursiveCharacterTextSplitter."""
        try:
            if not text.strip():
                api_logger.warning("Empty text provided for chunking")
                return []
            
            api_logger.debug(f"Chunking text of length {len(text)} with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
            
            # Use LangChain's RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_text(text)
            
            # Filter out empty chunks
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            api_logger.debug(f"Created {len(chunks)} chunks from text")
            
            # Log chunk statistics
            if chunks:
                avg_chunk_length = sum(len(chunk) for chunk in chunks) / len(chunks)
                api_logger.debug(f"Average chunk length: {avg_chunk_length:.1f} characters")
                api_logger.debug(f"Chunk length range: {min(len(chunk) for chunk in chunks)} - {max(len(chunk) for chunk in chunks)} characters")
            
            return chunks
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Error chunking text with LangChain",
                "Text chunking failed"
            )
            return []

    def process_paper(self, paper: Dict) -> bool:
        """Process a single paper with comprehensive error handling."""
        paper_title = paper.get('title', 'Unknown Title')
        
        with ErrorContext(f"processing_paper_{paper_title[:50]}", cleanup_func=None):
            try:
                api_logger.info(f"Processing paper: {paper_title}")
                
                # Validate paper data
                if not paper.get('pdf_url'):
                    api_logger.warning(f"No PDF URL found for paper: {paper_title}")
                    return False
                
                # Download PDF
                pdf_path = self.download_pdf(paper["pdf_url"])
                
                # Extract full text from PDF
                full_text = self.extract_text_from_pdf(pdf_path)
                
                if not full_text:
                    api_logger.warning(f"No text extracted from paper: {paper_title}")
                    os.remove(pdf_path)
                    return False
                
                # Chunk text using LangChain
                chunks = self.chunk_text_with_langchain(full_text)
                
                if not chunks:
                    api_logger.warning(f"No chunks created from paper: {paper_title}")
                    os.remove(pdf_path)
                    return False
                
                # Prepare metadata
                metadata = {
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "source": paper["pdf_url"],
                }
                
                # Store chunks
                section_metadata = [metadata for _ in chunks]
                self.retriever.store_chunks(chunks, metadata=section_metadata)
                
                # Cleanup
                os.remove(pdf_path)
                
                api_logger.info(f"Successfully processed paper: {paper_title} - Created {len(chunks)} chunks")
                return True
                
            except Exception as e:
                api_logger.log_error_with_recovery(
                    e,
                    f"Error processing paper: {paper_title}",
                    f"Paper processing failed: {str(e)}"
                )
                return False

    def run(self, query: str):
        """Search, download, chunk, and store results in the retriever."""
        try:
            api_logger.log_agent_activity("ResearchAgent", "run", query=query)
            
            # Search for papers
            papers = self.search(query)
            
            if not papers:
                api_logger.warning(f"No papers found for query: {query}")
                return []
            
            # Process each paper
            successful_papers = []
            failed_papers = []
            
            for paper in papers:
                success = self.process_paper(paper)
                if success:
                    successful_papers.append(paper)
                else:
                    failed_papers.append(paper)
            
            api_logger.info(f"Research completed. Success: {len(successful_papers)}, Failed: {len(failed_papers)}")
            
            if failed_papers:
                api_logger.warning(f"Failed to process {len(failed_papers)} papers")
                for paper in failed_papers:
                    api_logger.debug(f"Failed paper: {paper.get('title', 'Unknown')}")
            
            return successful_papers
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                f"Critical error in ResearchAgent.run for query: {query}",
                "Research workflow failed"
            )
            return []
