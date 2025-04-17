import argparse
import json
import os
from typing import Dict, List, Any, Tuple

# Import functions from scrape.py and parse.py
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    extract_mcq_content,
    split_dom_content,
    save_to_json
)
from parse import MCQProcessor

class MCQGenerator:
    def __init__(self, model_name="gemma:2b", host="http://localhost:11434"):
        self.processor = MCQProcessor(model_name=model_name, host=host)
        self.output_dir = "mcq_output"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_url(self, url: str, class_level: int = 12) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a URL to extract and format MCQs.
        
        Args:
            url (str): URL to process
            class_level (int): Class level (10, 11, or 12)
            
        Returns:
            Tuple: (formatted_mcqs, generated_mcqs)
        """
        print(f"Processing URL: {url}")
        
        # Scrape website
        html_content = scrape_website(url)
        
        if not html_content:
            print("Failed to get HTML content")
            return [], []
        
        # Try to extract MCQs directly from HTML structure
        mcq_data = extract_mcq_content(html_content)
        
        # If direct extraction failed, use AI to extract
        if not mcq_data:
            print("Direct MCQ extraction failed. Using AI extraction...")
            body_content = extract_body_content(html_content)
            cleaned_content = clean_body_content(body_content)
            
            # Split content into chunks
            chunks = split_dom_content(cleaned_content, max_length=4000)
            
            # Process chunks
            mcq_data = self.processor.process_chunks(chunks)
        
        if not mcq_data:
            print("No MCQs found on the page")
            return [], []
        
        # Format MCQs
        print(f"Formatting {len(mcq_data)} MCQs...")
        formatted_mcqs = self.processor.format_mcqs(mcq_data, class_level=class_level)
        
        # Save formatted MCQs
        output_file = os.path.join(self.output_dir, f"formatted_mcqs_{self._get_filename(url)}.json")
        self.processor.save_to_json(formatted_mcqs, output_file)
        
        # Generate additional MCQs on the same topic
        generated_mcqs = []
        if formatted_mcqs:
            # Extract topic from first MCQ
            topic = formatted_mcqs[0].get("topic", self._extract_topic_from_url(url))
            subject = formatted_mcqs[0].get("subject", self._extract_subject_from_url(url))
            
            print(f"Generating additional MCQs on topic: {topic}")
            generated_mcqs = self.processor.generate_mcqs(topic, count=5, class_level=class_level)
            
            if generated_mcqs:
                # Save generated MCQs
                output_file = os.path.join(self.output_dir, f"generated_mcqs_{self._get_filename(url)}.json")
                self.processor.save_to_json(generated_mcqs, output_file)
        
        return formatted_mcqs, generated_mcqs
    
    def batch_process(self, urls: List[str], class_level: int = 12) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple URLs in batch.
        
        Args:
            urls (List[str]): List of URLs to process
            class_level (int): Class level (10, 11, or 12)
            
        Returns:
            Dict: Dictionary mapping URLs to their processed MCQs
        """
        results = {}
        
        for url in urls:
            formatted_mcqs, generated_mcqs = self.process_url(url, class_level)
            results[url] = {
                "formatted": formatted_mcqs,
                "generated": generated_mcqs
            }
        
        # Save batch results
        output_file = os.path.join(self.output_dir, "batch_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _get_filename(self, url: str) -> str:
        """
        Generate a filename from a URL.
        
        Args:
            url (str): Input URL
            
        Returns:
            str: Sanitized filename
        """
        # Extract domain and path
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")
        path = parsed_url.path.strip("/").replace("/", "_")
        
        # Sanitize filename
        filename = f"{domain}_{path}"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        filename = filename[:100]  # Limit length
        
        return filename
    
    def _extract_topic_from_url(self, url: str) -> str:
        """
        Extract the topic from a URL.
        
        Args:
            url (str): Input URL
            
        Returns:
            str: Extracted topic
        """
        url_lower = url.lower()
        
        # Try to find keywords in the URL
        physics_topics = ["mechanics", "waves", "optics", "electricity", "magnetism", "modern-physics"]
        chemistry_topics = ["organic", "inorganic", "physical-chemistry", "periodic-table"]
        biology_topics = ["cell-biology", "genetics", "ecology", "human-physiology"]
        math_topics = ["algebra", "geometry", "calculus", "trigonometry", "statistics"]
        
        for topic in physics_topics + chemistry_topics + biology_topics + math_topics:
            if topic in url_lower:
                return topic.replace("-", " ").title()
        
        # Default topic
        return "General Science"
    
    def _extract_subject_from_url(self, url: str) -> str:
        """
        Extract the subject from a URL.
        
        Args:
            url (str): Input URL
            
        Returns:
            str: Extracted subject
        """
        url_lower = url.lower()
        
        if "physics" in url_lower:
            return "Physics"
        elif "chemistry" in url_lower:
            return "Chemistry"
        elif "biology" in url_lower:
            return "Biology"
        elif "math" in url_lower or "maths" in url_lower:
            return "Mathematics"
        
        # Default subject
        return "General Science"

def main():
    """
    Main function to run the MCQ Generator.
    """
    parser = argparse.ArgumentParser(description="CBSE MCQ Generator")
    parser.add_argument("--url", type=str, help="URL to scrape for MCQs")
    parser.add_argument("--urls-file", type=str, help="File containing URLs to scrape (one per line)")
    parser.add_argument("--class", dest="class_level", type=int, choices=[10, 11, 12], default=12, 
                       help="Class level (10, 11, or 12)")
    parser.add_argument("--model", type=str, default="gemma3:1b", 
                       help="Ollama model to use (default: gemma3:1b)")
    parser.add_argument("--host", type=str, default="http://localhost:11434", 
                       help="Ollama API host (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    if not args.url and not args.urls_file:
        parser.error("Either --url or --urls-file must be provided")
    
    # Initialize generator
    generator = MCQGenerator(model_name=args.model, host=args.host)
    
    if args.url:
        # Process single URL
        formatted_mcqs, generated_mcqs = generator.process_url(args.url, args.class_level)
        print(f"Processed {len(formatted_mcqs)} existing MCQs and generated {len(generated_mcqs)} new MCQs")
    
    if args.urls_file:
        # Process URLs from file
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(urls)} URLs from file")
        results = generator.batch_process(urls, args.class_level)
        
        # Print summary
        total_formatted = sum(len(result["formatted"]) for result in results.values())
        total_generated = sum(len(result["generated"]) for result in results.values())
        print(f"Processed {total_formatted} existing MCQs and generated {total_generated} new MCQs from {len(urls)} URLs")
    
    print("All processing complete. Results saved to 'mcq_output' directory.")

if __name__ == "__main__":
    main()