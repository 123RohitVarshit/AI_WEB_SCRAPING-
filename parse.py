from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from typing import List, Dict, Any

# Templates for different tasks
extract_template = """
You are an expert educational content processor specializing in CBSE board exams for classes 10-12.
You need to extract structured MCQ questions from the following HTML or text content: {content}

Instructions:
1. Identify all multiple-choice questions in the text.
2. For each question:
   - Extract the complete question text
   - Extract all answer options (typically labeled A, B, C, D)
   - Extract the correct answer if provided
   - Determine the subject area (Physics, Chemistry, Biology, or Mathematics)
   - Determine the class level (10, 11, or 12) based on the complexity
   
3. Format each question as a JSON object with the following structure:
{{
  "question": "Full question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "The correct answer or empty if not provided",
  "subject": "Subject area",
  "class": "10, 11, or 12",
  "topic": "The specific topic within the subject"
}}

4. If you can't find any MCQs, return an empty array [].
"""
   
format_template = """
You are an expert educational content creator specializing in CBSE board exam preparation.
Your task is to format and enhance the following MCQ questions for class {class_level} students:

{mcq_data}

Instructions:
1. Ensure each question is clear and follows proper CBSE exam format.
2. Make sure each question has exactly 4 options (A, B, C, D).
3. Enhance the questions by:
   - Adding proper scientific notation where relevant
   - Making sure units are correctly displayed
   - Adding diagrams where helpful (describe what diagram would be useful)
   - Ensuring each question tests a specific concept from the CBSE syllabus
4. For each question without an answer, provide the correct answer and a brief explanation.
5. Format the output as a well-structured JSON array.

Return the enhanced MCQs with the same structure but improved quality.
"""

generate_template = """
You are an expert educational content creator for CBSE board exams. 
Generate {count} new high-quality MCQ questions for class {class_level} students on the topic: {topic}.

Follow these requirements:
1. Each question must follow CBSE exam patterns and difficulty levels.
2. Each question must have 4 options (A, B, C, D).
3. One option must be correct, and three must be plausible distractors.
4. Include a variety of cognitive levels (knowledge, understanding, application, analysis).
5. Include the correct answer and a brief explanation for each question.
6. Make sure questions cover different aspects of the topic.
7. Format each question as a JSON object with the structure:
{{
  "question": "Full question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_option": "A, B, C, or D",
  "explanation": "Brief explanation of why this is the correct answer",
  "difficulty": "Easy, Medium, or Hard",
  "topic": "{topic}",
  "subject": "Physics/Chemistry/Biology/Mathematics",
  "class": "{class_level}"
}}

Return a JSON array of {count} well-structured MCQs.
"""

class MCQProcessor:
    def __init__(self, model_name="gemma:", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.model = OllamaLLM(model=model_name, base_url=host)  # Correct parameters

    def extract_mcqs(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract MCQs from raw content using LLM.
        
        Args:
            content (str): Raw HTML or text content
            
        Returns:
            List[Dict]: List of extracted MCQ data
        """
        prompt = ChatPromptTemplate.from_template(extract_template)
        chain = prompt | self.model
        
        try:
            response = chain.invoke({"content": content})
            response_text = response
            
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
                
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    print("No JSON array found in response")
                    return []
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {e}")
                return []
                
        except Exception as e:
            print(f"Error during MCQ extraction: {e}")
            return []
        
    def format_mcqs(self, mcq_data: List[Dict[str, Any]], class_level: int = 12) -> List[Dict[str, Any]]:
        """
        Format and enhance MCQs using LLM.
        
        Args:
            mcq_data (List[Dict]): List of MCQ data
            class_level (int): Class level (10, 11, or 12)
            
        Returns:
            List[Dict]: List of formatted MCQ data
        """
        if not mcq_data:
            return []
            
        prompt = ChatPromptTemplate.from_template(format_template)
        chain = prompt | self.model
        
        try:
            mcq_json = json.dumps(mcq_data, ensure_ascii=False)
            response = chain.invoke({
                "mcq_data": mcq_json,
                "class_level": class_level
            })
            
            response_text = response
            
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    print("No JSON array found in response")
                    return mcq_data
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {e}")
                return mcq_data
                
        except Exception as e:
            print(f"Error during MCQ formatting: {e}")
            return mcq_data
        
    def generate_mcqs(self, topic: str, count: int = 5, class_level: int = 12) -> List[Dict[str, Any]]:
        """
        Generate new MCQs on a specific topic using LLM.
        
        Args:
            topic (str): Topic to generate questions on
            count (int): Number of questions to generate
            class_level (int): Class level (10, 11, or 12)
            
        Returns:
            List[Dict]: List of generated MCQ data
        """
        prompt = ChatPromptTemplate.from_template(generate_template)
        chain = prompt | self.model
        
        try:
            response = chain.invoke({
                "topic": topic,
                "count": count,
                "class_level": class_level
            })
            
            response_text = response.content
            
            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    print("No JSON array found in response")
                    return []
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {e}")
                return []
                    
        except Exception as e:
            print(f"Error during MCQ generation: {e}")
            return []
        
    def process_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple content chunks and combine results.
        
        Args:
            chunks (List[str]): List of content chunks
            
        Returns:
            List[Dict]: Combined MCQ data from all chunks
        """
        all_mcqs = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            chunk_mcqs = self.extract_mcqs(chunk)
            all_mcqs.extend(chunk_mcqs)
            
        return all_mcqs
        
    def save_to_json(self, data: List[Dict[str, Any]], filename: str = "processed_mcqs.json"):
        """
        Save MCQ data to a JSON file.
        
        Args:
            data (List[Dict]): MCQ data to save
            filename (str): Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {filename}")

def main():
    """
    Main function to run the MCQ processing workflow.
    """
    from scrape import scrape_website, extract_body_content, clean_body_content, split_dom_content
    
    processor = MCQProcessor()
    
    # Example URL
    url = "https://www.studiestoday.com/mcq-physics-cbse-class-12-physics-electric-charges-and-fields-mcqs-set-291562.html"
    
    # Get content from website
    html_content = scrape_website(url)
    body_content = extract_body_content(html_content)
    cleaned_content = clean_body_content(body_content)
    
    # Split content into chunks
    chunks = split_dom_content(cleaned_content, max_length=4000)
    
    # Process chunks
    mcq_data = processor.process_chunks(chunks)
    
    # Format MCQs
    formatted_mcqs = processor.format_mcqs(mcq_data, class_level=12)
    
    # Save results
    processor.save_to_json(formatted_mcqs, "processed_mcqs.json")
    
    print(f"Processed {len(formatted_mcqs)} MCQ questions")
    
    # Generate additional MCQs on the same topic
    if formatted_mcqs:
        topic = formatted_mcqs[0].get("topic", "Electric Charges and Fields")
        generated_mcqs = processor.generate_mcqs(topic, count=3, class_level=12)
        
        if generated_mcqs:
            processor.save_to_json(generated_mcqs, "generated_mcqs.json")
            print(f"Generated {len(generated_mcqs)} additional MCQ questions")

if __name__ == "__main__":
    main()