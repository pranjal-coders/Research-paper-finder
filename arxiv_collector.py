import arxiv
import pandas as pd
import streamlit as st
import requests
from datetime import datetime
import time
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Clean and normalize text content"""

    text = re.sub(r'\s+', ' ', text)
    # Removing special characters that might cause issues
    text = text.strip()
    return text

def search_arxiv_papers(query: str, max_results: int = 200) -> List[Dict]:
    """
    Search arXiv for papers related to the query and return structured data
    """
    print(f"Searching arXiv for: {query}")
    print(f"Maximum results: {max_results}")
    client = arxiv.Client()
    
    # Create search query
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers_data = []
    
    try:
        for i, result in enumerate(client.results(search)):
            # Create paper data dictionary
            paper_data = {
                'title': clean_text(result.title),
                'summary': clean_text(result.summary),
                'authors': ', '.join([author.name for author in result.authors]),
                'published_date': result.published.strftime('%Y-%m-%d'),
                'updated_date': result.updated.strftime('%Y-%m-%d') if result.updated else '',
                'primary_category': result.primary_category,
                'categories': ', '.join(result.categories),
                'arxiv_id': result.entry_id.split('/')[-1],
                'pdf_url': result.pdf_url,
                'entry_url': result.entry_id,
                'doi': result.doi if result.doi else ''
            }
            
            papers_data.append(paper_data)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} papers...")
                
            # Add small delay to be respectful to arXiv API
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error during search: {e}")
        
    print(f"Successfully collected {len(papers_data)} papers")
    return papers_data

def create_dataset(papers_data: List[Dict]) -> pd.DataFrame:
    """
    Convert papers data to pandas DataFrame and save as CSV
    """
    df = pd.DataFrame(papers_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arxiv_ml_drone_papers_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Dataset saved as: {filename}")
    
    return df, filename

def main():
    """
    Main function to collect papers and create dataset
    """
    # Define search query for machine learning and adaptive behavior in drone systems
    query = """
    (machine learning OR deep learning OR reinforcement learning OR neural network) 
    AND (drone OR UAV OR "unmanned aerial vehicle" OR quadcopter OR multirotor) 
    AND (adaptive OR behavior OR control OR autonomous OR navigation)
    """
    
    # Collect papers
    papers_data = search_arxiv_papers(query, max_results=200)
    
    if papers_data:
        # Create dataset
        df, filename = create_dataset(papers_data)
        
        # Display basic statistics
        print(f"\nDataset Statistics:")
        print(f"Total papers: {len(df)}")
        print(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")
        print(f"Top categories:")
        category_counts = df['primary_category'].value_counts().head()
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")
            
        return df, filename
    else:
        print("No papers found!")
        return None, None

if __name__ == "__main__":
    main()